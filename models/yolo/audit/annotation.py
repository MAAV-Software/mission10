"""Loopback-only browser annotation server with explicit certification."""

from __future__ import annotations

import io
import json
import secrets
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .labels import (
    REVIEW_STATES,
    certify_labels,
    load_labels,
    resolve_source,
    sha256,
    validate_labels,
    write_labels,
)


ANNOTATION_HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission 10 real-image annotation</title>
<style>
body{font:15px system-ui;margin:0;background:#17191d;color:#eee}header,aside{padding:12px}
header{display:flex;gap:10px;align-items:center;background:#252831}main{display:grid;grid-template-columns:1fr 310px;gap:8px}
#stage{overflow:auto;text-align:center;padding:8px}canvas{max-width:100%;height:auto;cursor:crosshair;background:#000}
aside{background:#252831;min-height:calc(100vh - 64px)}button,select,input{margin:3px;padding:6px}
.record{padding:6px;border-bottom:1px solid #555}.mine{color:#ff6464}.ignore{color:#ffd166}
#status{margin-left:auto}.warning{color:#ffd166}
</style>
<header>
 <button id="prev">←</button><span id="identity"></span><button id="next">→</button>
 <span id="status"></span>
</header>
<main><section id="stage"><canvas id="canvas"></canvas></section><aside>
 <div><b>Draw</b>
  <select id="kind"><option value="mine">full-object mine box</option><option value="ignore">ignore region</option></select>
  <select id="visibility"><option>clear</option><option>partial</option><option>not_visible</option><option>unknown</option></select>
  <input id="reason" value="ambiguous" aria-label="ignore reason">
 </div>
 <p class="warning">Mine boxes cover the estimated full object, including hidden extent. Ignore regions are not negatives.</p>
 <div id="records"></div>
 <button id="save">Save annotation</button><button id="complete">Mark complete</button>
 <hr><b>Final human certification</b><br>
 <input id="reviewer" placeholder="reviewer name"><label><input id="ack" type="checkbox">I inspected every image</label><br>
 <button id="certify">Certify and lock labels</button>
</aside></main>
<script>
const token=new URLSearchParams(location.search).get('token');let doc,index=0,image,drag,dirty=false;
const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');
const api=(path,options={})=>fetch(path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(token),options).then(async r=>{if(!r.ok)throw Error(await r.text());return r.headers.get('content-type')?.includes('json')?r.json():r});
function boxOf(a,b){return [Math.min(a.x,b.x),Math.min(a.y,b.y),Math.max(a.x,b.x),Math.max(a.y,b.y)].map(v=>Math.round(v*100)/100)}
function point(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height}}
function draw(){ctx.drawImage(image,0,0);const rec=doc.images[index];
 for(const [kind,items,color] of [['mine',rec.objects,'#ff3030'],['ignore',rec.ignore_regions,'#ffd166']])for(const item of items){const b=item.xyxy;ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,canvas.width/500);ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
 if(drag){const b=boxOf(drag,drag.now);ctx.strokeStyle='#64d8ff';ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}}
function list(){const rec=doc.images[index];document.querySelector('#records').innerHTML=[...rec.objects.map((x,i)=>`<div class="record mine">mine ${x.xyxy.join(', ')} · ${x.visibility} <button data-k="objects" data-i="${i}">delete</button></div>`),...rec.ignore_regions.map((x,i)=>`<div class="record ignore">ignore ${x.xyxy.join(', ')} · ${x.reason} <button data-k="ignore_regions" data-i="${i}">delete</button></div>`)].join('');
 document.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{rec[b.dataset.k].splice(+b.dataset.i,1);rec.review_state='in_progress';dirty=true;list();draw()})}
async function show(){dirty=false;const rec=doc.images[index];document.querySelector('#identity').textContent=`${index+1}/${doc.images.length} ${rec.source} [${rec.role}] ${rec.review_state}`;image=new Image();image.onload=()=>{canvas.width=rec.width;canvas.height=rec.height;draw()};image.src=`/api/image/${index}?token=${encodeURIComponent(token)}&v=${Date.now()}`;list()}
async function save(state,refresh=true){const rec=doc.images[index];if(state)rec.review_state=state;doc=await api('/api/save',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({index,objects:rec.objects,ignore_regions:rec.ignore_regions,review_state:rec.review_state})});dirty=false;document.querySelector('#status').textContent='saved';if(refresh)show()}
canvas.onpointerdown=e=>{drag=point(e);drag.now=drag;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(drag){drag.now=point(e);draw()}};canvas.onpointerup=e=>{if(!drag)return;drag.now=point(e);const b=boxOf(drag,drag.now);drag=null;if(b[2]-b[0]<2||b[3]-b[1]<2)return draw();const rec=doc.images[index];if(document.querySelector('#kind').value==='mine')rec.objects.push({xyxy:b,visibility:document.querySelector('#visibility').value});else rec.ignore_regions.push({xyxy:b,reason:document.querySelector('#reason').value.trim()||'ambiguous'});rec.review_state='in_progress';dirty=true;list();draw()};
async function navigate(delta){if(dirty)await save(undefined,false);index=(index+doc.images.length+delta)%doc.images.length;show()}
document.querySelector('#prev').onclick=()=>navigate(-1);document.querySelector('#next').onclick=()=>navigate(1);document.querySelector('#save').onclick=()=>save();document.querySelector('#complete').onclick=()=>save('complete');
document.querySelector('#certify').onclick=async()=>{if(dirty)return alert('Save and mark this image complete before certification.');const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer||!document.querySelector('#ack').checked)return alert('Enter your name and acknowledge the final review.');if(!confirm('Certification locks every annotation. Continue?'))return;doc=await api('/api/certify',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({certified_by:reviewer,acknowledged:true})});alert('Labels certified and locked.');show()};
api('/api/document').then(x=>{doc=x;show()}).catch(e=>alert(e));
</script>"""


def _update_annotation(document: dict, update: object) -> dict:
    """Apply the only fields the browser is authorized to modify."""
    if "certification" in document:
        raise ValueError("certified labels are immutable")
    if not isinstance(update, dict) or not isinstance(update.get("index"), int):
        raise ValueError("annotation update requires integer index")
    index = update["index"]
    if not 0 <= index < len(document["images"]):
        raise ValueError("annotation index is out of range")
    if set(update) != {"index", "objects", "ignore_regions", "review_state"}:
        raise ValueError("annotation update contains unauthorized fields")
    if update["review_state"] not in REVIEW_STATES - {"certified"}:
        raise ValueError("browser cannot set certified review state")
    result = deepcopy(document)
    record = result["images"][index]
    record["objects"] = update["objects"]
    record["ignore_regions"] = update["ignore_regions"]
    record["review_state"] = update["review_state"]
    return validate_labels(result, require_frozen=True)


class AnnotationServer(ThreadingHTTPServer):
    labels_path: Path
    token: str


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationServer

    def _authorized(self) -> bool:
        query = parse_qs(urlparse(self.path).query)
        return secrets.compare_digest(query.get("token", [""])[0], self.server.token)

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self' 'unsafe-inline'; img-src 'self'",
        )
        self.end_headers()
        self.wfile.write(content)

    def _json(self, value: object, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(value).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._send(status, message.encode(), "text/plain; charset=utf-8")

    def _body(self) -> object:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if size < 1 or size > 10 * 1024 * 1024:
            raise ValueError("request body has invalid size")
        return json.loads(self.rfile.read(size))

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if not self._authorized():
            self._error(HTTPStatus.FORBIDDEN, "invalid annotation token")
            return
        try:
            if path == "/":
                self._send(HTTPStatus.OK, ANNOTATION_HTML.encode(), "text/html; charset=utf-8")
                return
            document = load_labels(self.server.labels_path, require_frozen=True)
            if path == "/api/document":
                self._json(document)
                return
            if path.startswith("/api/image/"):
                index = int(path.rsplit("/", 1)[1])
                record = document["images"][index]
                source_path = resolve_source(self.server.labels_path, record["source"])
                if sha256(source_path) != record["source_sha256"]:
                    raise ValueError("source image changed after annotation import")
                from PIL import Image, ImageOps

                with Image.open(source_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                if image.size != (record["width"], record["height"]):
                    raise ValueError("oriented image dimensions changed")
                output = io.BytesIO()
                image.save(output, "JPEG", quality=94)
                self._send(HTTPStatus.OK, output.getvalue(), "image/jpeg")
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, OSError, IndexError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if not self._authorized():
            self._error(HTTPStatus.FORBIDDEN, "invalid annotation token")
            return
        try:
            document = load_labels(self.server.labels_path, require_frozen=True)
            body = self._body()
            if path == "/api/save":
                document = _update_annotation(document, body)
            elif path == "/api/certify":
                if not isinstance(body, dict) or body.get("acknowledged") is not True:
                    raise ValueError("explicit certification acknowledgement required")
                document = certify_labels(document, str(body.get("certified_by", "")))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            write_labels(self.server.labels_path, document)
            self._json(document)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args) -> None:
        print(f"annotation: {format % args}")


def serve(labels_path: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the annotator on loopback. Non-loopback binding is forbidden."""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("annotation server must bind to loopback")
    load_labels(labels_path, require_frozen=True)
    server = AnnotationServer((host, port), AnnotationHandler)
    server.labels_path = labels_path.resolve()
    server.token = secrets.token_urlsafe(24)
    print(f"Open http://{host}:{server.server_port}/?token={server.token}")
    server.serve_forever()
