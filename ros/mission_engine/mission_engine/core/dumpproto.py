"""Regroup dump: payload builder + length-prefixed framing.

Socket-free by design — the ROS rim (or a laptop listener) moves the
bytes; everything here is pure encode/decode so the round-trip is unit-
testable. Schema is minefield-dump/1 (rfd-mission-execution, regroup
dump section); the master rejects unknown majors.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .minelog import MineLog

SCHEMA = "minefield-dump/1"
ACK = b"ok\n"
_MAX_FRAME = 16 * 1024 * 1024


def build_payload(
    log: MineLog,
    drone_id: str,
    mission_id: str,
    t_takeoff: float,
    t_dump: float,
    coverage: Optional[Dict] = None,
    stats: Optional[Dict] = None,
) -> Dict:
    log.finalize()
    mines: List[Dict] = []
    for c in log.clusters:
        mines.append(
            {
                "id": f"{drone_id}/{c.cluster_id}",
                "ll": list(c.ll) if c.ll is not None else None,
                "ll_per_pass": [list(p) for p in c.ll_per_pass],
                "local_xy": [c.centroid[0], c.centroid[1]],
                "spread_m": c.spread_m,
                "n_obs": c.n_obs,
                "n_passes": c.n_passes,
                "status": c.status,
                "tag_id": c.tag_ids[0] if c.tag_ids else None,
                "landmarks": [],
            }
        )
    return {
        "schema": SCHEMA,
        "drone_id": drone_id,
        "mission_id": mission_id,
        "t_takeoff": t_takeoff,
        "t_dump": t_dump,
        "mines": mines,
        "coverage": coverage if coverage is not None else {"lanes": [], "gaps": []},
        "stats": stats if stats is not None else {},
    }


def encode_frame(payload: Dict) -> bytes:
    """ASCII decimal length + newline, then the UTF-8 JSON document."""
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return str(len(body)).encode("ascii") + b"\n" + body


def decode_frame(data: bytes) -> Tuple[Dict, bytes]:
    """Parse one frame off the front of `data`; returns (payload, rest).
    Raises ValueError on a malformed frame or unknown schema major."""
    nl = data.find(b"\n")
    if nl < 0:
        raise ValueError("no length prefix")
    length = int(data[:nl])
    if not (0 < length <= _MAX_FRAME):
        raise ValueError(f"bad frame length {length}")
    body = data[nl + 1 : nl + 1 + length]
    if len(body) < length:
        raise ValueError(f"short frame: {len(body)} of {length} bytes")
    payload = json.loads(body.decode("utf-8"))
    schema = payload.get("schema", "")
    if schema.rsplit("/", 1)[0] != SCHEMA.rsplit("/", 1)[0] or (
        schema.rsplit("/", 1)[-1].split(".")[0] != SCHEMA.rsplit("/", 1)[-1]
    ):
        raise ValueError(f"unknown schema {schema!r}")
    return payload, data[nl + 1 + length :]
