"""Deterministic exact-source folds for certified private real imagery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .labels import load_labels, sha256, validate_labels


SCHEMA = "mission10-yolo-real-folds/1"
DEFAULT_FOLDS = 5
DEFAULT_ROLE = "training_candidate"
DEFAULT_SEED = "mission10-real-positive-folds-v1"
STRATA = ("clear", "partial", "negative")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stratum(record: dict) -> str:
    objects = [
        annotation
        for annotation in record["objects"]
        if annotation["visibility"] in {"clear", "partial"}
    ]
    if not objects:
        return "negative"
    if len(objects) != 1 or len(record["objects"]) != 1:
        raise ValueError(
            f"fold v1 requires zero or one scorable object: {record['source']}"
        )
    return objects[0]["visibility"]


def _order_key(seed: str, source_sha256: str) -> str:
    return hashlib.sha256(f"{seed}\0{source_sha256}".encode()).hexdigest()


def build_fold_document(
    labels: dict,
    labels_sha256: str,
    *,
    folds: int = DEFAULT_FOLDS,
    seed: str = DEFAULT_SEED,
    role: str = DEFAULT_ROLE,
) -> dict:
    """Return deterministic, stratum-balanced exact-photo assignments."""
    validate_labels(labels, require_frozen=True, require_certified=True)
    if folds < 2:
        raise ValueError("fold count must be at least two")
    if not isinstance(seed, str) or not seed:
        raise ValueError("fold seed must be non-empty")
    records = [record for record in labels["images"] if record["role"] == role]
    if not records:
        raise ValueError(f"no certified records assigned to role {role!r}")
    by_stratum = {stratum: [] for stratum in STRATA}
    for record in records:
        by_stratum[_stratum(record)].append(record)
    if any(not by_stratum[stratum] for stratum in STRATA):
        raise ValueError("fold input must contain clear, partial, and negative photos")

    assignments = []
    for stratum in STRATA:
        ordered = sorted(
            by_stratum[stratum],
            key=lambda record: (
                _order_key(seed, record["source_sha256"]),
                record["source_sha256"],
            ),
        )
        for ordinal, record in enumerate(ordered):
            assignments.append(
                {
                    "source": record["source"],
                    "source_sha256": record["source_sha256"],
                    "fold": ordinal % folds,
                    "stratum": stratum,
                    "objects": 0 if stratum == "negative" else 1,
                }
            )
    assignments.sort(key=lambda item: item["source_sha256"])
    counts = {
        str(fold): {
            "photos": sum(item["fold"] == fold for item in assignments),
            "clear": sum(
                item["fold"] == fold and item["stratum"] == "clear"
                for item in assignments
            ),
            "partial": sum(
                item["fold"] == fold and item["stratum"] == "partial"
                for item in assignments
            ),
            "negative": sum(
                item["fold"] == fold and item["stratum"] == "negative"
                for item in assignments
            ),
        }
        for fold in range(folds)
    }
    content = {
        "role": role,
        "grouping": "exact_source_photo",
        "seed": seed,
        "fold_count": folds,
        "assignments": assignments,
        "counts": counts,
    }
    return {
        "schema": SCHEMA,
        "labels_sha256": labels_sha256,
        "assignments_sha256": _canonical_sha256(assignments),
        **content,
    }


def validate_fold_document(
    document: object,
    *,
    labels: dict | None = None,
    labels_sha256: str | None = None,
) -> dict:
    """Validate a fold lock and optionally bind it to certified labels."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"fold lock must use schema {SCHEMA}")
    folds = document.get("fold_count")
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
        raise ValueError("fold_count must be an integer of at least two")
    if document.get("grouping") != "exact_source_photo":
        raise ValueError("fold lock must group by exact source photo")
    if not isinstance(document.get("seed"), str) or not document["seed"]:
        raise ValueError("fold lock seed must be non-empty")
    assignments = document.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("fold lock assignments must be a non-empty list")
    hashes = set()
    sources = set()
    for index, item in enumerate(assignments):
        where = f"assignments[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        source = item.get("source")
        source_hash = item.get("source_sha256")
        fold = item.get("fold")
        stratum = item.get("stratum")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{where}.source must be non-empty")
        if source in sources:
            raise ValueError(f"duplicate fold source {source!r}")
        sources.add(source)
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
            or source_hash in hashes
        ):
            raise ValueError(f"{where}.source_sha256 is invalid or duplicate")
        hashes.add(source_hash)
        if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < folds:
            raise ValueError(f"{where}.fold is outside the fold range")
        if stratum not in STRATA:
            raise ValueError(f"{where}.stratum is invalid")
        expected_objects = 0 if stratum == "negative" else 1
        if item.get("objects") != expected_objects:
            raise ValueError(f"{where}.objects disagrees with stratum")
    if document.get("assignments_sha256") != _canonical_sha256(assignments):
        raise ValueError("fold assignments changed after locking")
    if labels_sha256 is not None and document.get("labels_sha256") != labels_sha256:
        raise ValueError("fold lock labels hash changed")
    if labels is not None:
        validate_labels(labels, require_frozen=True, require_certified=True)
        expected = {
            record["source_sha256"]: record
            for record in labels["images"]
            if record["role"] == document.get("role")
        }
        if set(expected) != hashes:
            raise ValueError("fold assignments do not exactly cover the selected role")
        for item in assignments:
            record = expected[item["source_sha256"]]
            if item["source"] != record["source"] or item["stratum"] != _stratum(
                record
            ):
                raise ValueError(f"fold assignment changed for {record['source']}")
    expected_counts = {
        str(fold): {
            "photos": sum(item["fold"] == fold for item in assignments),
            "clear": sum(
                item["fold"] == fold and item["stratum"] == "clear"
                for item in assignments
            ),
            "partial": sum(
                item["fold"] == fold and item["stratum"] == "partial"
                for item in assignments
            ),
            "negative": sum(
                item["fold"] == fold and item["stratum"] == "negative"
                for item in assignments
            ),
        }
        for fold in range(folds)
    }
    if document.get("counts") != expected_counts:
        raise ValueError("fold counts disagree with assignments")
    return document


def load_fold_document(path: Path, labels_path: Path | None = None) -> dict:
    """Load a fold lock and optionally revalidate its certified labels."""
    document = json.loads(Path(path).read_text())
    if labels_path is None:
        return validate_fold_document(document)
    labels = load_labels(labels_path, require_frozen=True, require_certified=True)
    return validate_fold_document(
        document, labels=labels, labels_sha256=sha256(labels_path)
    )


def source_hashes(document: dict, fold: int, *, held_out: bool = True) -> set[str]:
    """Return source hashes on one side of a fold split."""
    validate_fold_document(document)
    if (
        not isinstance(fold, int)
        or isinstance(fold, bool)
        or not 0 <= fold < document["fold_count"]
    ):
        raise ValueError("held-out fold is outside the fold range")
    return {
        item["source_sha256"]
        for item in document["assignments"]
        if (item["fold"] == fold) == held_out
    }


def select_records(records: Iterable[dict], document: dict, fold: int) -> list[dict]:
    """Select held-out records from an already validated fold lock."""
    selected = source_hashes(document, fold)
    result = [record for record in records if record["source_sha256"] in selected]
    if {record["source_sha256"] for record in result} != selected:
        raise ValueError("fold selection is missing certified records")
    return result
