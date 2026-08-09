"""Loopback-only browser review for SAM mine cutout proposals."""

from __future__ import annotations

import json
import os
import secrets
import threading
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .masks import load_mask_review, validate_mask_review


def counts(review: dict) -> dict:
    result = {
        "total": len(review["entries"]),
        "pending": 0,
        "confirmed": 0,
        "rejected": 0,
    }
    for entry in review["entries"]:
        result[entry["confirmation"]] += 1
    return result


def update_confirmation(review: dict, payload: object) -> dict:
    validate_mask_review(review)
    if not isinstance(payload, dict) or set(payload) != {"id", "confirmation"}:
        raise ValueError("review update may contain only id and confirmation")
    if payload["confirmation"] not in {"confirmed", "rejected"}:
        raise ValueError("confirmation must be confirmed or rejected")
    matches = [entry for entry in review["entries"] if entry["id"] == payload["id"]]
    if len(matches) != 1:
        raise ValueError("review entry id was not found")
    result = deepcopy(review)
    next(entry for entry in result["entries"] if entry["id"] == payload["id"])[
        "confirmation"
    ] = payload["confirmation"]
    return validate_mask_review(result)


def _write(path: Path, review: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(review, indent=2) + "\n")
    os.replace(temporary, path)


class ReviewSession:
    def __init__(self, review_path: Path):
        self.path = review_path.resolve()
        self.root = self.path.parent
        self.lock = threading.RLock()
        self.review = load_mask_review(self.path)

    def state(self) -> dict:
        with self.lock:
            return {"counts": counts(self.review), "entries": self.review["entries"]}

    def update(self, payload: object) -> dict:
        with self.lock:
            current = load_mask_review(self.path)
            updated = update_confirmation(current, payload)
            _write(self.path, updated)
            self.review = updated
            return self.state()

    def preview(self, entry_id: str) -> bytes:
        with self.lock:
            matches = [
                entry for entry in self.review["entries"] if entry["id"] == entry_id
            ]
            if len(matches) != 1:
                raise ValueError("review entry id was not found")
            path = (self.root / matches[0]["preview"]).resolve()
            if self.root not in path.parents:
                raise ValueError("preview escapes review directory")
            return path.read_bytes()


HTML = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission 10 SAM mask review</title><style>body{font:16px system-ui;margin:0;background:#15171b;color:#eee;text-align:center}header{padding:12px;background:#242832;position:sticky;top:0}main{max-width:1050px;margin:auto;padding:16px}img{max-width:100%;max-height:72vh;background:#000;border:1px solid #666}button{font-size:18px;padding:10px 20px;margin:8px}.yes{background:#19723e;color:#fff}.no{background:#922f39;color:#fff}.meta{white-space:pre-line;line-height:1.5}</style>
<header><button id="prev">←</button><span id="counts"></span><button id="next">→</button></header><main><h2 id="progress"></h2><img id="preview"><div class="meta" id="meta"></div><button class="yes" id="yes">Y Confirm mask</button><button class="no" id="no">N Reject mask</button><div id="notice"></div></main><script>
const token=new URLSearchParams(location.search).get('token');let s,pos=0,busy=false;const api=(p,o={})=>fetch(p+(p.includes('?')?'&':'?')+'token='+encodeURIComponent(token),o).then(async r=>{if(!r.ok)throw Error(await r.text());return r.json()});function render(){const a=s.entries,c=s.counts;document.querySelector('#counts').textContent=`${c.pending} pending · ${c.confirmed} confirmed · ${c.rejected} rejected`;pos=Math.min(pos,a.length-1);const e=a[pos];document.querySelector('#progress').textContent=`${pos+1}/${a.length}`;document.querySelector('#preview').src=`/api/preview/${e.id}?token=${encodeURIComponent(token)}`;document.querySelector('#meta').textContent=`${e.source}\nprompt (yellow): ${e.prompt_xyxy.join(', ')}\nSAM mask (cyan) · ${e.segmentation_pixels} px · ${e.confirmation}`}function nav(d){pos=(pos+s.entries.length+d)%s.entries.length;render()}async function decide(x){if(busy)return;busy=true;const e=s.entries[pos];document.querySelector('#notice').textContent='saving…';try{s=await api('/api/confirmation',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id:e.id,confirmation:x})});document.querySelector('#notice').textContent=`saved ${x}`;pos=(pos+1)%s.entries.length}catch(e){alert(e)}finally{busy=false;render()}}document.querySelector('#prev').onclick=()=>nav(-1);document.querySelector('#next').onclick=()=>nav(1);document.querySelector('#yes').onclick=()=>decide('confirmed');document.querySelector('#no').onclick=()=>decide('rejected');document.onkeydown=e=>{if(e.key==='ArrowLeft')nav(-1);else if(e.key==='ArrowRight')nav(1);else if(e.key.toLowerCase()==='y')decide('confirmed');else if(e.key.toLowerCase()==='n')decide('rejected')};api('/api/state').then(x=>{s=x;render()}).catch(alert)</script>"""


class Server(ThreadingHTTPServer):
    session: ReviewSession
    token: str


class Handler(BaseHTTPRequestHandler):
    server: Server

    def authorized(self):
        return secrets.compare_digest(
            parse_qs(urlparse(self.path).query).get("token", [""])[0], self.server.token
        )

    def send_content(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if not self.authorized():
            return self.send_content(
                HTTPStatus.FORBIDDEN, b"invalid token", "text/plain"
            )
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self.send_content(HTTPStatus.OK, HTML.encode(), "text/html")
            if path == "/api/state":
                return self.send_content(
                    HTTPStatus.OK,
                    json.dumps(self.server.session.state()).encode(),
                    "application/json",
                )
            if path.startswith("/api/preview/"):
                return self.send_content(
                    HTTPStatus.OK,
                    self.server.session.preview(path.rsplit("/", 1)[1]),
                    "image/png",
                )
            self.send_content(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        except (ValueError, OSError) as error:
            self.send_content(HTTPStatus.BAD_REQUEST, str(error).encode(), "text/plain")

    def do_POST(self):  # noqa: N802
        if not self.authorized():
            return self.send_content(
                HTTPStatus.FORBIDDEN, b"invalid token", "text/plain"
            )
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            state = self.server.session.update(payload)
            self.send_content(
                HTTPStatus.OK, json.dumps(state).encode(), "application/json"
            )
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self.send_content(HTTPStatus.BAD_REQUEST, str(error).encode(), "text/plain")

    def log_message(self, format, *args):
        print(f"mask-review: {format % args}")


def serve(review_path: Path, *, host: str = "127.0.0.1", port: int = 8767) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("mask review server must bind to loopback")
    server = Server((host, port), Handler)
    server.session = ReviewSession(review_path)
    server.token = secrets.token_urlsafe(24)
    print(f"Open http://{host}:{server.server_port}/?token={server.token}")
    server.serve_forever()
