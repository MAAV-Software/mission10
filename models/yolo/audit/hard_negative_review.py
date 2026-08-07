"""Loopback-only browser review for proposed real-image hard negatives."""

from __future__ import annotations

import io
import json
import os
import secrets
import threading
from collections import OrderedDict
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .hard_negatives import revalidate_review, validate_review
from .labels import resolve_source, sha256


DECISIONS = frozenset({"confirmed", "rejected"})


def confirmation_counts(review: dict) -> dict[str, int]:
    """Return stable pending/decision counts for a valid review document."""
    validate_review(review)
    counts = {
        "total": len(review["entries"]),
        "pending": 0,
        "confirmed": 0,
        "rejected": 0,
    }
    for entry in review["entries"]:
        counts[entry["confirmation"]] += 1
    return counts


def display_indices(review: dict, labels: dict) -> list[int]:
    """Group crops by manifest source without changing stored proposal order."""
    validate_review(review)
    source_order = {
        record["source_sha256"]: index
        for index, record in enumerate(labels["images"])
    }

    def key(index: int) -> tuple:
        entry = review["entries"][index]
        confidence = entry["baseline_confidence"]
        return (
            source_order.get(entry["source_sha256"], len(source_order)),
            entry["source"],
            0 if entry["kind"] == "baseline_candidate" else 1,
            -float(confidence) if confidence is not None else 0.0,
            entry["tile_xyxy"][1],
            entry["tile_xyxy"][0],
            entry["id"],
        )

    return sorted(range(len(review["entries"])), key=key)


def resume_position(review: dict, order: list[int]) -> int | None:
    """Return the first pending position in display order, or the first item."""
    validate_review(review)
    if (
        len(set(order)) != len(order)
        or any(index < 0 or index >= len(review["entries"]) for index in order)
    ):
        raise ValueError("display order must contain valid, unique review indices")
    for position, index in enumerate(order):
        if review["entries"][index]["confirmation"] == "pending":
            return position
    return 0 if order else None


def _evenly_spaced(indices: list[int], count: int) -> list[int]:
    """Select deterministic confidence/source coverage across an ordered pool."""
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)
    return [indices[(2 * rank + 1) * len(indices) // (2 * count)] for rank in range(count)]


def qa_display_indices(review: dict, labels: dict, qa_size: int | None) -> list[int]:
    """Build a compact, stable QA sample while retaining prior decisions.

    Certification, rather than this spot-check, is the authority for negative
    labels. The sample preserves every decision already made and spreads new
    items across source photos, detector confidence, and clean controls.
    """
    full_order = display_indices(review, labels)
    if qa_size is None or qa_size >= len(full_order):
        return full_order
    if qa_size < 1:
        raise ValueError("QA size must be positive")

    decided = [
        index
        for index in full_order
        if review["entries"][index]["confirmation"] != "pending"
    ]
    target = max(qa_size, len(decided))
    needed = target - len(decided)
    if needed == 0:
        return decided

    excluded = set(decided)
    covered_sources = {
        review["entries"][index]["source_sha256"] for index in decided
    }

    def one_per_source(kind: str, *, avoid: set[str]) -> list[int]:
        by_source: dict[str, list[int]] = {}
        for index in full_order:
            entry = review["entries"][index]
            if (
                index in excluded
                or entry["kind"] != kind
                or entry["source_sha256"] in avoid
            ):
                continue
            by_source.setdefault(entry["source_sha256"], []).append(index)
        representatives = []
        for source_hash, source_indices in by_source.items():
            if kind == "baseline_candidate":
                representatives.append(
                    max(
                        source_indices,
                        key=lambda index: (
                            review["entries"][index]["baseline_confidence"],
                            review["entries"][index]["id"],
                        ),
                    )
                )
            else:
                representatives.append(
                    min(source_indices, key=lambda index: review["entries"][index]["id"])
                )
        if kind == "baseline_candidate":
            representatives.sort(
                key=lambda index: (
                    review["entries"][index]["baseline_confidence"],
                    review["entries"][index]["source_sha256"],
                )
            )
        else:
            representatives.sort(key=lambda index: review["entries"][index]["source_sha256"])
        return representatives

    baseline_target = min(needed, max(1, round(needed * 0.75)))
    baseline_pool = one_per_source("baseline_candidate", avoid=covered_sources)
    selected = _evenly_spaced(baseline_pool, baseline_target)
    selected_sources = {
        review["entries"][index]["source_sha256"] for index in selected
    }
    clean_pool = one_per_source(
        "deterministic_clean", avoid=covered_sources | selected_sources
    )
    selected.extend(_evenly_spaced(clean_pool, needed - len(selected)))

    if len(selected) < needed:
        already = excluded | set(selected)
        fallback = [index for index in full_order if index not in already]
        selected.extend(fallback[: needed - len(selected)])
    chosen = excluded | set(selected)
    return [index for index in full_order if index in chosen]


def update_confirmation(review: dict, update: object) -> dict:
    """Apply the browser's sole authorized mutation to a deep copy."""
    validate_review(review)
    if not isinstance(update, dict) or set(update) != {"id", "confirmation"}:
        raise ValueError("review update may contain only id and confirmation")
    entry_id = update["id"]
    decision = update["confirmation"]
    if not isinstance(entry_id, str) or not entry_id:
        raise ValueError("review update requires an entry id")
    if decision not in DECISIONS:
        raise ValueError("confirmation must be confirmed or rejected")
    matches = [
        index for index, entry in enumerate(review["entries"])
        if entry["id"] == entry_id
    ]
    if len(matches) != 1:
        raise ValueError("review entry id was not found")
    result = deepcopy(review)
    result["entries"][matches[0]]["confirmation"] = decision
    return validate_review(result)


def _write_review(path: Path, review: dict) -> None:
    """Replace a validated review atomically in its existing directory."""
    validate_review(review)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(review, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ReviewSession:
    """Validated review state with serialized, confirmation-only updates."""

    def __init__(
        self,
        review_path: Path,
        labels_path: Path,
        baseline_path: Path,
        *,
        qa_size: int | None = 32,
    ):
        self.review_path = review_path.resolve()
        self.labels_path = labels_path.resolve()
        self.baseline_path = baseline_path.resolve()
        self._lock = threading.RLock()
        review, labels = self._load()
        self._review = review
        self._labels = labels
        self._verify_source_hashes()
        self._display_order = qa_display_indices(review, labels, qa_size)

    def _load(self) -> tuple[dict, dict]:
        review = validate_review(json.loads(self.review_path.read_text()))
        labels = revalidate_review(review, self.labels_path, self.baseline_path)
        return review, labels

    def _verify_source_hashes(self) -> None:
        records = {
            record["source_sha256"]: record for record in self._labels["images"]
        }
        checked: set[str] = set()
        for entry in self._review["entries"]:
            source_hash = entry["source_sha256"]
            if source_hash in checked:
                continue
            record = records[source_hash]
            source_path = resolve_source(self.labels_path, record["source"])
            if sha256(source_path) != source_hash:
                raise ValueError(f"source changed: {source_path}")
            checked.add(source_hash)

    def _state_unlocked(self) -> dict:
        order = self._display_order
        position = resume_position(self._review, order)
        entries = []
        for index in order:
            entry = self._review["entries"][index]
            entries.append(
                {
                    "id": entry["id"],
                    "source": entry["source"],
                    "tile_xyxy": entry["tile_xyxy"],
                    "kind": entry["kind"],
                    "baseline_confidence": entry["baseline_confidence"],
                    "confirmation": entry["confirmation"],
                }
            )
        qa_counts = {
            "total": len(order),
            "pending": 0,
            "confirmed": 0,
            "rejected": 0,
        }
        for index in order:
            qa_counts[self._review["entries"][index]["confirmation"]] += 1
        return {
            "entries": entries,
            "counts": qa_counts,
            "pool_counts": confirmation_counts(self._review),
            "resume_id": entries[position]["id"] if position is not None else None,
            "tile_px": self._review["tile_px"],
        }

    def state(self) -> dict:
        with self._lock:
            return self._state_unlocked()

    def update(self, payload: object) -> dict:
        with self._lock:
            # Reload and revalidate immediately before every write. This catches
            # external provenance, baseline, label, or certification changes.
            review, labels = self._load()
            updated = update_confirmation(review, payload)
            revalidate_review(updated, self.labels_path, self.baseline_path)
            if updated != review:
                _write_review(self.review_path, updated)
            self._review = updated
            self._labels = labels
            return self._state_unlocked()

    def crop(self, entry_id: str, cache: "SourceImageCache") -> bytes:
        with self._lock:
            matches = [
                entry for entry in self._review["entries"]
                if entry["id"] == entry_id
            ]
            if len(matches) != 1:
                raise ValueError("review entry id was not found")
            entry = matches[0]
            record = next(
                record for record in self._labels["images"]
                if record["source_sha256"] == entry["source_sha256"]
            )
        crop = cache.crop(self.labels_path, record, entry["tile_xyxy"])
        output = io.BytesIO()
        crop.save(output, "PNG", optimize=False)
        return output.getvalue()


class SourceImageCache:
    """Small LRU of fully decoded EXIF-oriented sources for adjacent crops."""

    def __init__(self, max_sources: int = 1):
        if max_sources < 1:
            raise ValueError("source cache must retain at least one image")
        self.max_sources = max_sources
        self._images: OrderedDict[str, tuple[tuple[int, int], object]] = OrderedDict()
        self._lock = threading.Lock()

    def crop(self, labels_path: Path, record: dict, rect: list[int]):
        from PIL import Image, ImageOps

        source_path = resolve_source(labels_path, record["source"])
        stat = source_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        key = str(source_path)
        with self._lock:
            cached = self._images.get(key)
            if cached is None or cached[0] != signature:
                if sha256(source_path) != record["source_sha256"]:
                    raise ValueError(f"source changed: {source_path}")
                with Image.open(source_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                if image.size != (record["width"], record["height"]):
                    image.close()
                    raise ValueError(f"oriented dimensions changed: {source_path}")
                if cached is not None:
                    cached[1].close()
                self._images[key] = (signature, image)
            else:
                self._images.move_to_end(key)
                image = cached[1]
            while len(self._images) > self.max_sources:
                _, (_, evicted) = self._images.popitem(last=False)
                evicted.close()
            # Pillow pads an edge tile outside the oriented source exactly as
            # materialize() does, retaining the deployment-size 640 px crop.
            return image.crop(tuple(rect))


REVIEW_HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission 10 hard-negative review</title>
<style>
body{font:16px system-ui;margin:0;background:#15171b;color:#f2f2f2}header{display:flex;gap:12px;align-items:center;padding:12px 18px;background:#242832;position:sticky;top:0}main{max-width:980px;margin:auto;padding:14px;text-align:center}img{width:min(640px,100%);height:auto;image-rendering:auto;background:#000;border:1px solid #555}.counts{margin-left:auto}.metadata{line-height:1.6;margin:10px;white-space:pre-line}.decision{font-size:18px;padding:10px 22px;margin:8px}.yes{background:#196b3b;color:white}.no{background:#8b2933;color:white}button,select{padding:7px}kbd{background:#333;padding:2px 5px;border-radius:3px}#notice{min-height:1.4em;color:#8be9fd}
</style>
<header><button id="prev">← Prev</button><button id="next">Next →</button>
<label>Show <select id="filter"><option value="pending">pending</option><option value="all">all</option></select></label>
<span class="counts" id="counts"></span></header>
<main><h2 id="progress"></h2><div id="empty" hidden>All proposed crops have decisions.</div>
<img id="crop" alt="exact EXIF-oriented source crop"><div class="metadata" id="metadata"></div>
<button class="decision yes" id="confirm"><kbd>Y</kbd> Confirm negative</button>
<button class="decision no" id="reject"><kbd>N</kbd> Reject</button>
<div><kbd>←</kbd>/<kbd>→</kbd> navigate · decisions save immediately</div><div id="notice"></div></main>
<script>
const token=new URLSearchParams(location.search).get('token');let state,shown=[],position=0,busy=false,displayedId=null;
const api=(path,options={})=>fetch(path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(token),options).then(async r=>{if(!r.ok)throw Error(await r.text());return r.json()});
function filtered(){return state.entries.filter(e=>document.querySelector('#filter').value==='all'||e.confirmation==='pending')}
function render(preferred){shown=filtered();if(preferred){const found=shown.findIndex(e=>e.id===preferred);if(found>=0)position=found}if(position>=shown.length)position=Math.max(0,shown.length-1);const c=state.counts,p=state.pool_counts;document.querySelector('#counts').textContent=`QA: ${c.pending} pending · ${c.confirmed} confirmed · ${c.rejected} rejected · ${c.total} sampled (${p.total} certified candidates)`;const empty=!shown.length;document.querySelector('#empty').hidden=!empty;document.querySelector('#crop').hidden=empty;document.querySelector('#confirm').disabled=empty||busy;document.querySelector('#reject').disabled=empty||busy;if(empty){document.querySelector('#progress').textContent='QA sample complete';document.querySelector('#metadata').textContent='';return}const e=shown[position];document.querySelector('#progress').textContent=`${position+1}/${shown.length} shown · ${c.total-c.pending}/${c.total} QA decisions`;if(displayedId!==e.id){displayedId=e.id;document.querySelector('#crop').src=`/api/crop/${e.id}?token=${encodeURIComponent(token)}`};const conf=e.baseline_confidence===null?'none (deterministic clean)':e.baseline_confidence.toFixed(4);document.querySelector('#metadata').textContent=`${e.source}\n${e.kind} · confidence ${conf}\ncrop ${e.tile_xyxy.join(', ')} · ${e.confirmation}`}
function navigate(delta){if(!shown.length)return;position=(position+shown.length+delta)%shown.length;render(shown[position].id)}
async function decide(confirmation){if(busy||!shown.length)return;busy=true;const oldPosition=position,current=shown[position];document.querySelector('#notice').textContent='saving…';render(current.id);try{state=await api('/api/confirmation',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({id:current.id,confirmation})});shown=filtered();position=document.querySelector('#filter').value==='pending'?Math.min(oldPosition,Math.max(0,shown.length-1)):(oldPosition+1)%Math.max(1,shown.length);document.querySelector('#notice').textContent=`saved ${confirmation}`;}catch(e){document.querySelector('#notice').textContent=e.message;alert(e)}finally{busy=false;render()}}
document.querySelector('#prev').onclick=()=>navigate(-1);document.querySelector('#next').onclick=()=>navigate(1);document.querySelector('#confirm').onclick=()=>decide('confirmed');document.querySelector('#reject').onclick=()=>decide('rejected');document.querySelector('#filter').onchange=()=>{position=0;render(state.resume_id)};document.onkeydown=e=>{if(e.repeat)return;if(e.key==='ArrowLeft')navigate(-1);else if(e.key==='ArrowRight')navigate(1);else if(e.key.toLowerCase()==='y')decide('confirmed');else if(e.key.toLowerCase()==='n')decide('rejected')};
api('/api/state').then(x=>{state=x;render(state.resume_id)}).catch(e=>alert(e));
</script>"""


class HardNegativeReviewServer(ThreadingHTTPServer):
    session: ReviewSession
    token: str
    image_cache: SourceImageCache


class HardNegativeReviewHandler(BaseHTTPRequestHandler):
    server: HardNegativeReviewServer

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
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, value: object) -> None:
        self._send(HTTPStatus.OK, json.dumps(value).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._send(status, message.encode(), "text/plain; charset=utf-8")

    def _body(self) -> object:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if size < 1 or size > 64 * 1024:
            raise ValueError("request body has invalid size")
        return json.loads(self.rfile.read(size))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not self._authorized():
            self._error(HTTPStatus.FORBIDDEN, "invalid review token")
            return
        try:
            if path == "/":
                self._send(HTTPStatus.OK, REVIEW_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(self.server.session.state())
            elif path.startswith("/api/crop/"):
                entry_id = path.rsplit("/", 1)[1]
                content = self.server.session.crop(entry_id, self.server.image_cache)
                self._send(HTTPStatus.OK, content, "image/png")
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, OSError, StopIteration) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not self._authorized():
            self._error(HTTPStatus.FORBIDDEN, "invalid review token")
            return
        try:
            if path != "/api/confirmation":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._json(self.server.session.update(self._body()))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args) -> None:
        print(f"hard-negative-review: {format % args}")


def serve(
    review_path: Path,
    labels_path: Path,
    baseline_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    qa_size: int | None = 32,
) -> None:
    """Serve the reviewer on loopback; non-loopback binding is forbidden."""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("hard-negative review server must bind to loopback")
    session = ReviewSession(
        review_path,
        labels_path,
        baseline_path,
        qa_size=qa_size,
    )
    server = HardNegativeReviewServer((host, port), HardNegativeReviewHandler)
    server.session = session
    server.token = secrets.token_urlsafe(24)
    server.image_cache = SourceImageCache(max_sources=1)
    print(f"Open http://{host}:{server.server_port}/?token={server.token}")
    server.serve_forever()
