"""Schema and integrity checks for Mission 10 real-image annotations.

Coordinates are pixel ``[x0, y0, x1, y1]`` boxes in the EXIF-oriented image.
Mine boxes describe the estimated *full object*, including its hidden extent.
``visibility`` records how much of that full object can actually be seen.

Roles are assigned by capture group and then frozen before annotation:

``training_candidate``
    May be considered for training after review. It is never automatically
    included in training.
``development_eval``
    Immutable development evaluation data (including the audited phone and
    legacy synthetic images).
``final_holdout``
    CM2 final evaluation data. Evaluation requires certified labels.
``ood_holdout``
    Monochrome/out-of-distribution evaluation data.
``hard_negative_pool``
    Optional, explicitly training-eligible hard-negative-only captures.

Review states are ``unreviewed``, ``in_progress``, ``complete``, and
``certified``. Certification is a deliberate human action; model output never
advances a review state.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


LABEL_SCHEMA = "mission10-yolo-real-labels/1"
ROLES = frozenset(
    {
        "training_candidate",
        "development_eval",
        "final_holdout",
        "ood_holdout",
        "hard_negative_pool",
    }
)
EVALUATION_ROLES = frozenset(
    {"development_eval", "final_holdout", "ood_holdout"}
)
TRAINING_ROLES = frozenset({"training_candidate", "hard_negative_pool"})
REVIEW_STATES = frozenset(
    {"unreviewed", "in_progress", "complete", "certified"}
)
VISIBILITIES = frozenset({"clear", "partial", "not_visible", "unknown"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _role_assignments(document: Mapping) -> list[dict]:
    return [
        {
            "source_sha256": image["source_sha256"],
            "capture_group": image["capture_group"],
            "role": image["role"],
        }
        for image in sorted(
            document["images"], key=lambda record: record["source_sha256"]
        )
    ]


def _annotation_payload(document: Mapping) -> dict:
    payload = deepcopy(dict(document))
    certification = payload.pop("certification", None)
    if certification is not None:
        payload["certification"] = {
            key: certification.get(key) for key in ("certified_by", "certified_at")
        }
    return payload


def _box(value: object, width: int, height: int, where: str) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{where}: xyxy must contain four numbers")
    x0, y0, x1, y1 = map(float, value)
    if not (0.0 <= x0 < x1 <= width and 0.0 <= y0 < y1 <= height):
        raise ValueError(f"{where}: xyxy is outside {width}x{height}: {value!r}")
    return x0, y0, x1, y1


def validate_labels(
    document: object,
    *,
    require_frozen: bool = False,
    require_certified: bool = False,
) -> dict:
    """Validate and return ``document`` without silently normalizing it."""
    if not isinstance(document, dict):
        raise ValueError("label document must be an object")
    if document.get("schema") != LABEL_SCHEMA:
        raise ValueError(
            f"expected schema {LABEL_SCHEMA}, got {document.get('schema')!r}"
        )
    images = document.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty list")

    sources: set[str] = set()
    hashes: set[str] = set()
    group_roles: dict[str, str] = {}
    for image_index, image in enumerate(images):
        where = f"images[{image_index}]"
        if not isinstance(image, dict):
            raise ValueError(f"{where}: expected object")
        source = image.get("source")
        source_hash = image.get("source_sha256")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{where}: source must be a non-empty string")
        if source in sources:
            raise ValueError(f"{where}: duplicate source {source!r}")
        sources.add(source)
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError(f"{where}: source_sha256 must be lowercase SHA-256")
        if source_hash in hashes:
            raise ValueError(f"{where}: duplicate source content {source_hash}")
        hashes.add(source_hash)

        width, height = image.get("width"), image.get("height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width < 1
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height < 1
        ):
            raise ValueError(f"{where}: width and height must be positive integers")
        capture_group = image.get("capture_group")
        role = image.get("role")
        if not isinstance(capture_group, str) or not capture_group.strip():
            raise ValueError(f"{where}: capture_group must be non-empty")
        if role not in ROLES:
            raise ValueError(f"{where}: invalid role {role!r}")
        previous_role = group_roles.setdefault(capture_group, role)
        if previous_role != role:
            raise ValueError(
                f"capture group {capture_group!r} has both {previous_role!r} and {role!r}"
            )
        if image.get("review_state") not in REVIEW_STATES:
            raise ValueError(f"{where}: invalid review_state")

        objects = image.get("objects")
        regions = image.get("ignore_regions")
        if not isinstance(objects, list) or not isinstance(regions, list):
            raise ValueError(f"{where}: objects and ignore_regions must be lists")
        for object_index, annotation in enumerate(objects):
            annotation_where = f"{where}.objects[{object_index}]"
            if not isinstance(annotation, dict):
                raise ValueError(f"{annotation_where}: expected object")
            _box(annotation.get("xyxy"), width, height, annotation_where)
            if annotation.get("visibility") not in VISIBILITIES:
                raise ValueError(f"{annotation_where}: invalid visibility")
        for region_index, region in enumerate(regions):
            region_where = f"{where}.ignore_regions[{region_index}]"
            if not isinstance(region, dict):
                raise ValueError(f"{region_where}: expected object")
            _box(region.get("xyxy"), width, height, region_where)
            reason = region.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{region_where}: reason must be non-empty")

    freeze = document.get("role_freeze")
    if freeze is not None:
        if not isinstance(freeze, dict) or freeze.get("state") != "frozen":
            raise ValueError("role_freeze must have state 'frozen'")
        assignments = _role_assignments(document)
        if freeze.get("assignments_sha256") != canonical_sha256(assignments):
            raise ValueError("capture-group roles changed after role freeze")
        if not isinstance(freeze.get("frozen_by"), str) or not freeze["frozen_by"].strip():
            raise ValueError("role_freeze.frozen_by must be non-empty")
    elif require_frozen:
        raise ValueError("capture-group roles must be frozen")

    certification = document.get("certification")
    if certification is not None:
        if not isinstance(certification, dict):
            raise ValueError("certification must be an object")
        if any(image["review_state"] != "certified" for image in images):
            raise ValueError("all images must be certified in a certified document")
        expected = canonical_sha256(_annotation_payload(document))
        if certification.get("annotations_sha256") != expected:
            raise ValueError("annotations changed after human certification")
        if not isinstance(certification.get("certified_by"), str) or not certification[
            "certified_by"
        ].strip():
            raise ValueError("certification.certified_by must be non-empty")
    elif require_certified:
        raise ValueError("labels require final human certification")
    return document


def freeze_roles(document: dict, frozen_by: str, *, now: str | None = None) -> dict:
    """Freeze capture-group roles and return a modified deep copy."""
    if not frozen_by.strip():
        raise ValueError("frozen_by must be non-empty")
    result = deepcopy(document)
    validate_labels(result)
    if "role_freeze" in result:
        raise ValueError("capture-group roles are already frozen")
    assignments = _role_assignments(result)
    result["role_freeze"] = {
        "state": "frozen",
        "frozen_by": frozen_by.strip(),
        "frozen_at": now or datetime.now(timezone.utc).isoformat(),
        "assignments_sha256": canonical_sha256(assignments),
    }
    validate_labels(result, require_frozen=True)
    return result


def certify_labels(document: dict, certified_by: str, *, now: str | None = None) -> dict:
    """Perform the final, explicit human certification step."""
    if not certified_by.strip():
        raise ValueError("certified_by must be non-empty")
    result = deepcopy(document)
    validate_labels(result, require_frozen=True)
    if "certification" in result:
        raise ValueError("labels are already certified")
    incomplete = [
        image["source"]
        for image in result["images"]
        if image["review_state"] not in {"complete", "certified"}
    ]
    if incomplete:
        raise ValueError(f"cannot certify incomplete images: {incomplete}")
    for image in result["images"]:
        image["review_state"] = "certified"
    result["certification"] = {
        "certified_by": certified_by.strip(),
        "certified_at": now or datetime.now(timezone.utc).isoformat(),
    }
    result["certification"]["annotations_sha256"] = canonical_sha256(
        _annotation_payload(result)
    )
    validate_labels(result, require_frozen=True, require_certified=True)
    return result


def load_labels(
    path: Path, *, require_frozen: bool = False, require_certified: bool = False
) -> dict:
    document = json.loads(path.read_text())
    return validate_labels(
        document,
        require_frozen=require_frozen,
        require_certified=require_certified,
    )


def write_labels(path: Path, document: dict) -> None:
    validate_labels(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)


def new_labels(
    paths: Iterable[Path],
    *,
    base: Path,
    capture_group: str,
    role: str,
) -> dict:
    """Create unreviewed records with EXIF-oriented dimensions.

    Pillow is imported only here, keeping schema/evaluation metric tests free
    from image-runtime dependencies.
    """
    if role not in ROLES:
        raise ValueError(f"invalid role {role!r}")
    if not capture_group.strip():
        raise ValueError("capture_group must be non-empty")
    from PIL import Image, ImageOps

    images = []
    for raw_path in sorted({Path(path).resolve() for path in paths}):
        with Image.open(raw_path) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
        try:
            source_name = str(raw_path.relative_to(base.resolve()))
        except ValueError:
            source_name = str(raw_path)
        images.append(
            {
                "source": source_name,
                "source_sha256": sha256(raw_path),
                "width": width,
                "height": height,
                "capture_group": capture_group,
                "role": role,
                "review_state": "unreviewed",
                "objects": [],
                "ignore_regions": [],
            }
        )
    document = {"schema": LABEL_SCHEMA, "images": images}
    return validate_labels(document)


def resolve_source(labels_path: Path, source: str) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = labels_path.resolve().parent / path
    return path.resolve()
