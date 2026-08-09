"""SAM cutout proposals for certified, fully clear real mines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .labels import load_labels, resolve_source, sha256


SCHEMA = "mission10-yolo-mask-review/1"
DECISIONS = frozenset({"pending", "confirmed", "rejected"})


def _entry_id(source_hash: str, object_index: int) -> str:
    return hashlib.sha256(f"{source_hash}\0{object_index}".encode()).hexdigest()[:20]


def validate_mask_review(document: object) -> dict:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"mask review must use schema {SCHEMA}")
    if not isinstance(document.get("labels_sha256"), str):
        raise ValueError("mask review lacks labels_sha256")
    if not isinstance(document.get("sam_weights_sha256"), str):
        raise ValueError("mask review lacks sam_weights_sha256")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("mask review entries must be a non-empty list")
    ids = set()
    sources = set()
    for index, entry in enumerate(entries):
        where = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object")
        if not isinstance(entry.get("id"), str) or entry["id"] in ids:
            raise ValueError(f"{where}.id is invalid or duplicate")
        ids.add(entry["id"])
        if entry.get("visibility") != "clear":
            raise ValueError(f"{where} must be a clear mine")
        source_key = (entry.get("source_sha256"), entry.get("object_index"))
        if source_key in sources:
            raise ValueError(f"{where} duplicates a source object")
        sources.add(source_key)
        if entry.get("confirmation") not in DECISIONS:
            raise ValueError(f"{where}.confirmation is invalid")
        for name in ("prompt_xyxy", "segmentation_xyxy"):
            box = entry.get(name)
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(not isinstance(value, int) for value in box)
                or not (box[0] < box[2] and box[1] < box[3])
            ):
                raise ValueError(f"{where}.{name} is invalid")
        for name in ("cutout", "preview", "cutout_sha256", "preview_sha256"):
            if not isinstance(entry.get(name), str) or not entry[name]:
                raise ValueError(f"{where}.{name} must be non-empty")
    return document


def load_mask_review(path: Path, *, require_decided: bool = False) -> dict:
    document = validate_mask_review(json.loads(Path(path).read_text()))
    if require_decided and any(
        entry["confirmation"] == "pending" for entry in document["entries"]
    ):
        raise ValueError("mask review still has pending proposals")
    return document


def _mask_array(result, expected_size: tuple[int, int]):
    import numpy as np

    if result.masks is None or len(result.masks.data) != 1:
        raise ValueError("SAM must return exactly one mask for one box prompt")
    data = result.masks.data[0]
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    mask = np.asarray(data) > 0.5
    if mask.shape != (expected_size[1], expected_size[0]):
        from PIL import Image

        mask = (
            np.asarray(
                Image.fromarray(mask.astype("uint8") * 255).resize(
                    expected_size, Image.Resampling.NEAREST
                )
            )
            > 0
        )
    return mask


def _predict(model, image, box: list[int], device: str | None):
    kwargs = {
        "source": image,
        "bboxes": [box],
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    results = list(model.predict(**kwargs))
    if len(results) != 1:
        raise ValueError("SAM must return exactly one result per source")
    return _mask_array(results[0], image.size)


def _save_artifacts(image, mask, box, entry_id: str, out: Path) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("SAM returned an empty mask")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    alpha = Image.fromarray(mask.astype("uint8") * 255, "L")
    cutout = image.crop((x0, y0, x1, y1)).convert("RGBA")
    cutout.putalpha(alpha.crop((x0, y0, x1, y1)))
    cutout_path = out / "cutouts" / f"{entry_id}.png"
    cutout.save(cutout_path, "PNG", optimize=False)

    bx0, by0, bx1, by1 = box
    padding = max(bx1 - bx0, by1 - by0)
    preview_box = (
        max(0, bx0 - padding),
        max(0, by0 - padding),
        min(image.width, bx1 + padding),
        min(image.height, by1 + padding),
    )
    preview = image.crop(preview_box).convert("RGBA")
    local_mask = alpha.crop(preview_box)
    tint = Image.new("RGBA", preview.size, (0, 255, 255, 115))
    preview.alpha_composite(
        Image.composite(tint, Image.new("RGBA", preview.size), local_mask)
    )
    draw = ImageDraw.Draw(preview)
    draw.rectangle(
        (
            bx0 - preview_box[0],
            by0 - preview_box[1],
            bx1 - preview_box[0],
            by1 - preview_box[1],
        ),
        outline=(255, 230, 0, 255),
        width=max(2, round(min(preview.size) / 150)),
    )
    if max(preview.size) < 900:
        factor = min(6, max(1, 900 // max(preview.size)))
        preview = preview.resize(
            (preview.width * factor, preview.height * factor),
            Image.Resampling.NEAREST,
        )
    preview_path = out / "previews" / f"{entry_id}.png"
    preview.convert("RGB").save(preview_path, "PNG", optimize=False)
    return {
        "segmentation_xyxy": [x0, y0, x1, y1],
        "segmentation_pixels": int(mask.sum()),
        "cutout": str(Path("cutouts") / cutout_path.name),
        "cutout_sha256": sha256(cutout_path),
        "preview": str(Path("previews") / preview_path.name),
        "preview_sha256": sha256(preview_path),
    }


def propose_masks(
    labels_path: Path,
    sam_weights: Path,
    out: Path,
    *,
    device: str | None = None,
    model=None,
) -> dict:
    """Propose one box-prompted SAM cutout for every certified clear mine."""
    labels_path, sam_weights, out = map(Path, (labels_path, sam_weights, out))
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    labels = load_labels(labels_path, require_frozen=True, require_certified=True)
    if not sam_weights.is_file():
        raise ValueError(f"SAM weights do not exist: {sam_weights}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cutouts").mkdir(exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    if model is None:
        from ultralytics import SAM

        model = SAM(str(sam_weights))
    from PIL import Image, ImageOps

    entries = []
    for record in labels["images"]:
        source_path = resolve_source(labels_path, record["source"])
        clear = [
            (index, annotation)
            for index, annotation in enumerate(record["objects"])
            if annotation["visibility"] == "clear"
        ]
        if not clear:
            continue
        if sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source hash changed: {source_path}")
        with Image.open(source_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"oriented dimensions changed: {source_path}")
        for object_index, annotation in clear:
            box = [round(value) for value in annotation["xyxy"]]
            proposal_id = _entry_id(record["source_sha256"], object_index)
            mask = _predict(model, image, box, device)
            artifacts = _save_artifacts(image, mask, box, proposal_id, out)
            entries.append(
                {
                    "id": proposal_id,
                    "source": record["source"],
                    "source_sha256": record["source_sha256"],
                    "object_index": object_index,
                    "visibility": "clear",
                    "prompt_xyxy": box,
                    **artifacts,
                    "confirmation": "pending",
                }
            )
    entries.sort(key=lambda entry: (entry["source_sha256"], entry["object_index"]))
    if not entries:
        raise ValueError("certified labels contain no clear mines")
    document = {
        "schema": SCHEMA,
        "labels": str(labels_path.resolve()),
        "labels_sha256": sha256(labels_path),
        "sam_weights": str(sam_weights.resolve()),
        "sam_weights_sha256": sha256(sam_weights),
        "proposal_method": "sam_box_prompt",
        "policy": {
            "cutout_visibility": "clear_only",
            "partial_definition": "any_occluding_grass_blade_is_partial",
            "partial_use": "whole_context_positive_only",
            "human_confirmation_required": True,
        },
        "entries": entries,
    }
    validate_mask_review(document)
    (out / "mask-review.json").write_text(json.dumps(document, indent=2) + "\n")
    return document
