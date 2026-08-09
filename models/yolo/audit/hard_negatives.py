"""Conservative, certification-backed hard-negative materialization."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from .evaluation import Box, intersection
from .labels import (
    TRAINING_ROLES,
    canonical_sha256,
    load_labels,
    resolve_source,
    sha256,
)


REVIEW_SCHEMA = "mission10-yolo-hard-negative-review/1"
COMPONENT_SCHEMA = "mission10-yolo-training-component/1"
BASELINE_MIN_CONFIDENCE = 0.10
MAX_BASELINE_PER_PHOTO = 8
CLEAN_PER_PHOTO = 2
CONFIRMATION_STATES = frozenset({"pending", "confirmed", "rejected"})


def _rect_intersects_annotations(rect: Box, record: dict) -> bool:
    annotations = [item["xyxy"] for item in record["objects"]]
    annotations.extend(item["xyxy"] for item in record["ignore_regions"])
    return any(intersection(rect, Box.from_xyxy(xyxy)) > 0.0 for xyxy in annotations)


def _tile_rect(x: int, y: int, tile: int, width: int, height: int) -> Box:
    # Only real source pixels participate in intersection checks. Pillow may
    # later pad the right/bottom edge to retain an exact deployment tile.
    return Box(
        float(x),
        float(y),
        float(min(x + tile, width)),
        float(min(y + tile, height)),
    )


def _tile_origins(
    width: int, height: int, tile: int, overlap: int
) -> list[tuple[int, int]]:
    # Keep this pure module testable without importing ROS. This intentionally
    # mirrors mission_engine.core.tiles.tile_grid's even-spacing rule.
    def axis(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        count = 1 + math.ceil((extent - tile) / (tile - overlap))
        span = extent - tile
        return [round(index * span / (count - 1)) for index in range(count)]

    return [(x, y) for y in axis(height) for x in axis(width)]


def _candidate_id(source_hash: str, x: int, y: int, tile: int) -> str:
    return canonical_sha256(
        {"source_sha256": source_hash, "tile_xyxy": [x, y, x + tile, y + tile]}
    )[:24]


def _audit_records(audit: dict) -> dict[str, dict]:
    if audit.get("schema") != "mission10-yolo-irl-audit/1":
        raise ValueError("baseline must be a mission10-yolo-irl-audit/1 report")
    result = {}
    for record in audit.get("images", []):
        source_hash = record.get("source_sha256")
        if not isinstance(source_hash, str) or source_hash in result:
            raise ValueError("baseline has missing or duplicate source hashes")
        result[source_hash] = record
    return result


def _proposal_entry(
    record: dict,
    *,
    x: int,
    y: int,
    tile: int,
    kind: str,
    confidence: float | None,
) -> dict:
    return {
        "id": _candidate_id(record["source_sha256"], x, y, tile),
        "source": record["source"],
        "source_sha256": record["source_sha256"],
        "capture_group": record["capture_group"],
        "role": record["role"],
        "tile_xyxy": [x, y, x + tile, y + tile],
        "kind": kind,
        "baseline_confidence": confidence,
        "confirmation": "pending",
    }


def propose(
    labels_path: Path,
    baseline_path: Path,
) -> dict:
    """Create a deterministic pending-review proposal; write no training data."""
    document = load_labels(
        labels_path, require_frozen=True, require_certified=True
    )
    baseline = json.loads(baseline_path.read_text())
    baseline_by_hash = _audit_records(baseline)
    tile = baseline.get("tile_px")
    overlap = baseline.get("overlap_px")
    if (
        not isinstance(tile, int)
        or tile < 1
        or not isinstance(overlap, int)
        or not 0 <= overlap < tile
    ):
        raise ValueError("baseline has invalid tile geometry")
    if baseline.get("candidate_floor", 1.0) > BASELINE_MIN_CONFIDENCE:
        raise ValueError("baseline did not preserve candidates down to 0.10")

    entries = []
    for record in document["images"]:
        if record["role"] not in TRAINING_ROLES:
            continue
        if record["review_state"] != "certified":
            continue
        audit_record = baseline_by_hash.get(record["source_sha256"])
        if audit_record is None:
            raise ValueError(f"baseline is missing eligible source {record['source']}")
        if "candidates" not in audit_record:
            raise ValueError("baseline lacks 0.001 candidates; rerun tools/audit_irl.py")
        if (
            audit_record.get("width") != record["width"]
            or audit_record.get("height") != record["height"]
        ):
            raise ValueError(f"baseline dimensions disagree for {record['source']}")

        # Select at most one proposal per source tile, using its strongest
        # >=0.10 candidate. Stable coordinate tie-breaks make review repeatable.
        strongest: dict[tuple[int, int], float] = {}
        valid_origins = set(
            _tile_origins(record["width"], record["height"], tile, overlap)
        )
        for candidate in audit_record["candidates"]:
            confidence = float(candidate["confidence"])
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"baseline has invalid confidence for {record['source']}")
            if confidence < BASELINE_MIN_CONFIDENCE:
                continue
            origin = (int(candidate["tile_x"]), int(candidate["tile_y"]))
            if origin not in valid_origins:
                raise ValueError(f"baseline has invalid tile origin for {record['source']}")
            strongest[origin] = max(confidence, strongest.get(origin, -1.0))
        baseline_origins = []
        for (x, y), confidence in sorted(
            strongest.items(), key=lambda item: (-item[1], item[0][1], item[0][0])
        ):
            rect = _tile_rect(x, y, tile, record["width"], record["height"])
            if _rect_intersects_annotations(rect, record):
                continue
            baseline_origins.append((x, y, confidence))
            if len(baseline_origins) == MAX_BASELINE_PER_PHOTO:
                break

        used = {(x, y) for x, y, _ in baseline_origins}
        for x, y, confidence in baseline_origins:
            entries.append(
                _proposal_entry(
                    record,
                    x=x,
                    y=y,
                    tile=tile,
                    kind="baseline_candidate",
                    confidence=confidence,
                )
            )

        clean_origins = []
        for x, y in sorted(
            _tile_origins(record["width"], record["height"], tile, overlap),
            key=lambda origin: canonical_sha256(
                {"source_sha256": record["source_sha256"], "origin": origin}
            ),
        ):
            if (x, y) in used:
                continue
            rect = _tile_rect(x, y, tile, record["width"], record["height"])
            if _rect_intersects_annotations(rect, record):
                continue
            clean_origins.append((x, y))
            if len(clean_origins) == CLEAN_PER_PHOTO:
                break
        for x, y in clean_origins:
            entries.append(
                _proposal_entry(
                    record,
                    x=x,
                    y=y,
                    tile=tile,
                    kind="deterministic_clean",
                    confidence=None,
                )
            )

    entries.sort(key=lambda entry: entry["id"])
    proposal = {
        "schema": REVIEW_SCHEMA,
        "labels": str(labels_path.resolve()),
        "labels_sha256": sha256(labels_path),
        "baseline": str(baseline_path.resolve()),
        "baseline_sha256": sha256(baseline_path),
        "policy": {
            "baseline_min_confidence": BASELINE_MIN_CONFIDENCE,
            "max_baseline_per_photo": MAX_BASELINE_PER_PHOTO,
            "deterministic_clean_per_photo": CLEAN_PER_PHOTO,
            "mine_or_ignore_intersection_allowed": False,
            "eligible_roles": sorted(TRAINING_ROLES),
            "eligible_review_states": ["certified"],
        },
        "tile_px": tile,
        "overlap_px": overlap,
        "entries": entries,
    }
    validate_review(proposal)
    return proposal


def validate_review(review: object, *, require_decided: bool = False) -> dict:
    if not isinstance(review, dict) or review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"expected {REVIEW_SCHEMA}")
    entries = review.get("entries")
    if not isinstance(entries, list):
        raise ValueError("review entries must be a list")
    tile = review.get("tile_px")
    overlap = review.get("overlap_px")
    if (
        not isinstance(tile, int)
        or isinstance(tile, bool)
        or tile < 1
        or not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or not 0 <= overlap < tile
    ):
        raise ValueError("review has invalid tile geometry")
    seen = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entries[{index}] must be an object")
        if entry.get("id") in seen:
            raise ValueError(f"duplicate review entry {entry.get('id')!r}")
        seen.add(entry.get("id"))
        if entry.get("confirmation") not in CONFIRMATION_STATES:
            raise ValueError(f"entries[{index}] has invalid confirmation")
        if require_decided and entry["confirmation"] == "pending":
            raise ValueError(f"hard-negative review is incomplete: {entry['id']}")
        if entry.get("kind") not in {"baseline_candidate", "deterministic_clean"}:
            raise ValueError(f"entries[{index}] has invalid kind")
        source_hash = entry.get("source_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError(f"entries[{index}] has invalid source hash")
        rect = entry.get("tile_xyxy")
        if (
            not isinstance(rect, list)
            or len(rect) != 4
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in rect
            )
            or rect[2] - rect[0] != tile
            or rect[3] - rect[1] != tile
        ):
            raise ValueError(f"entries[{index}] has invalid tile")
        if entry.get("id") != _candidate_id(source_hash, rect[0], rect[1], tile):
            raise ValueError(f"entries[{index}] identity does not match provenance")
        confidence = entry.get("baseline_confidence")
        if entry["kind"] == "baseline_candidate" and (
            not isinstance(confidence, (int, float))
            or confidence < BASELINE_MIN_CONFIDENCE
        ):
            raise ValueError(f"entries[{index}] has invalid baseline confidence")
        if entry["kind"] == "deterministic_clean" and confidence is not None:
            raise ValueError(f"entries[{index}] clean tile has model confidence")
    return review


def revalidate_review(
    review: dict, labels_path: Path, baseline_path: Path
) -> dict:
    """Revalidate review provenance and return its certified label document.

    A human review is authorized to change only entry ``confirmation`` values.
    Everything else is reproduced from the certified labels and frozen baseline
    so callers cannot accidentally turn the review file into a provenance edit
    or a certification bypass.
    """
    validate_review(review)
    if sha256(labels_path) != review.get("labels_sha256"):
        raise ValueError("labels changed after hard-negative proposal")
    if sha256(baseline_path) != review.get("baseline_sha256"):
        raise ValueError("baseline changed after hard-negative proposal")
    expected = propose(labels_path, baseline_path)

    actual_provenance = deepcopy(review)
    expected_provenance = deepcopy(expected)
    for entry in actual_provenance["entries"]:
        entry.pop("confirmation")
    for entry in expected_provenance["entries"]:
        entry.pop("confirmation")
    if actual_provenance != expected_provenance:
        raise ValueError("hard-negative proposal provenance was modified")
    document = load_labels(
        labels_path, require_frozen=True, require_certified=True
    )
    records = {record["source_sha256"]: record for record in document["images"]}
    for entry in review["entries"]:
        record = records.get(entry["source_sha256"])
        if record is None or record["source"] != entry["source"]:
            raise ValueError(f"review source changed: {entry['id']}")
        if record["role"] not in TRAINING_ROLES or record["review_state"] != "certified":
            raise ValueError(f"review source is no longer training-eligible: {entry['id']}")
        x0, y0, _, _ = entry["tile_xyxy"]
        visible_rect = _tile_rect(
            x0, y0, review["tile_px"], record["width"], record["height"]
        )
        if _rect_intersects_annotations(visible_rect, record):
            raise ValueError(f"review tile intersects a mine or ignore region: {entry['id']}")
    return document


def _review_against_sources(review: dict, labels_path: Path, baseline_path: Path) -> dict:
    """Backward-compatible private alias for the public provenance validator."""
    return revalidate_review(review, labels_path, baseline_path)


def materialize(
    review_path: Path,
    labels_path: Path,
    baseline_path: Path,
    out: Path,
    *,
    certification_backed: bool = False,
) -> dict:
    """Materialize revalidated, unique negative train tiles.

    By default every proposal still requires an explicit human decision.  With
    ``certification_backed=True``, a pending proposal may also be included
    because the source is exhaustively annotated, human-certified, hash-locked,
    and the tile is rechecked to intersect neither a mine nor an ignore region.
    A human rejection always excludes a tile.
    """
    review = validate_review(
        json.loads(review_path.read_text()),
        require_decided=not certification_backed,
    )
    document = _review_against_sources(review, labels_path, baseline_path)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    image_out = out / "images" / "train"
    label_out = out / "labels" / "train"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    records = {record["source_sha256"]: record for record in document["images"]}

    from PIL import Image, ImageOps

    lock_entries = []
    duplicates = []
    image_owner: dict[str, str] = {}
    selection_basis_counts = {
        "human_confirmed": 0,
        "certified_annotation_absence": 0,
        "human_rejected": 0,
    }
    for entry in review["entries"]:
        if entry["confirmation"] == "rejected":
            selection_basis_counts["human_rejected"] += 1
            continue
        if entry["confirmation"] == "confirmed":
            selection_basis = "human_confirmed"
        elif certification_backed:
            selection_basis = "certified_annotation_absence"
        else:  # require_decided above makes this unreachable.
            continue
        record = records[entry["source_sha256"]]
        source_path = resolve_source(labels_path, record["source"])
        if sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source changed: {source_path}")
        with Image.open(source_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"oriented dimensions changed: {source_path}")
        tile = image.crop(tuple(entry["tile_xyxy"]))
        tile_name = f"hn_{entry['id']}"
        image_path = image_out / f"{tile_name}.png"
        tile.save(image_path, "PNG", optimize=False)
        image_hash = sha256(image_path)
        owner = image_owner.get(image_hash)
        if owner is not None:
            image_path.unlink()
            duplicates.append({"entry_id": entry["id"], "duplicate_of": owner})
            continue
        image_owner[image_hash] = entry["id"]
        selection_basis_counts[selection_basis] += 1
        label_path = label_out / f"{tile_name}.txt"
        label_path.write_text("")
        lock_entries.append(
            {
                "tile": tile_name,
                "boxes": 0,
                "image_sha256": image_hash,
                "label_sha256": sha256(label_path),
                "provenance": {
                    "review_entry_id": entry["id"],
                    "review_confirmation": entry["confirmation"],
                    "selection_basis": selection_basis,
                    "kind": entry["kind"],
                    "baseline_confidence": entry["baseline_confidence"],
                    "source": entry["source"],
                    "source_sha256": entry["source_sha256"],
                    "capture_group": entry["capture_group"],
                    "tile_xyxy": entry["tile_xyxy"],
                },
            }
        )
    lock_entries.sort(key=lambda item: item["tile"])
    if not lock_entries:
        raise ValueError("review contains no confirmed, unique hard negatives")
    content = {"entries": {"train": lock_entries}}
    lock = {
        "schema": COMPONENT_SCHEMA,
        "component": (
            "certification-backed-real-hard-negatives"
            if certification_backed
            else "human-confirmed-real-hard-negatives"
        ),
        "scope": "train_only",
        "labels_sha256": sha256(labels_path),
        "baseline_sha256": sha256(baseline_path),
        "review_sha256": sha256(review_path),
        "policy": deepcopy(review["policy"]),
        **content,
        "duplicates": duplicates,
        "selection_basis_counts": selection_basis_counts,
        "counts": {"train": {"tiles": len(lock_entries), "boxes": 0}},
        "dataset_sha256": canonical_sha256(content),
    }
    (out / "component.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock
