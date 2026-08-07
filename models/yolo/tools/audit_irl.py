"""Run deployment-style tiled YOLO inference on unlabeled IRL images.

This is a qualitative domain-gap audit. It preserves each original image's
identity, applies EXIF orientation, uses the shared 640 px inference grid, and
writes full-resolution overlays plus machine-readable detections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

MISSION_ENGINE_SRC = Path(__file__).resolve().parents[3] / "ros" / "mission_engine"
if str(MISSION_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(MISSION_ENGINE_SRC))

from mission_engine.core.tiles import tile_grid  # noqa: E402


SCHEMA = "mission10-yolo-irl-audit/1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
CANDIDATE_FLOOR = 0.001


@dataclass(frozen=True)
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    tile_x: int
    tile_y: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _area(box: Detection) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def _intersection(first: Detection, second: Detection) -> float:
    return max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0)) * max(
        0.0, min(first.y1, second.y1) - max(first.y0, second.y0)
    )


def _duplicate_overlap(first: Detection, second: Detection) -> float:
    """Return the larger of IoU and overlap over the smaller box.

    A detection clipped by a tile boundary can cover less than half of the
    complete detection and evade ordinary IoU NMS. Overlap over the smaller
    box identifies that pair without relying on tile-local coordinates.
    """
    intersection = _intersection(first, second)
    first_area = _area(first)
    second_area = _area(second)
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    smaller = min(first_area, second_area)
    containment = intersection / smaller if smaller > 0.0 else 0.0
    return max(iou, containment)


def merge_detections(
    detections: Sequence[Detection], overlap_threshold: float = 0.5
) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(
        detections, key=lambda detection: detection.confidence, reverse=True
    ):
        if all(
            _duplicate_overlap(candidate, accepted) < overlap_threshold
            for accepted in kept
        ):
            kept.append(candidate)
    return kept


def _paths(inputs: Iterable[Path]) -> list[Path]:
    paths = []
    for source in inputs:
        if source.is_dir():
            paths.extend(
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(source)
        else:
            raise ValueError(f"not a supported image or directory: {source}")
    return sorted(set(path.resolve() for path in paths))


def _predict_chunks(
    model, crops: Sequence, batch: int, tile: int, device: str | None
):
    for start in range(0, len(crops), batch):
        yield from model.predict(
            source=list(crops[start : start + batch]),
            imgsz=tile,
            conf=CANDIDATE_FLOOR,
            iou=0.7,
            max_det=100,
            batch=batch,
            device=device,
            verbose=False,
            stream=False,
        )


def _draw_overlay(image, detections: Sequence[Detection]):
    from PIL import ImageDraw, ImageFont

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    width = max(3, round(min(image.size) / 500))
    for detection in detections:
        coordinates = (detection.x0, detection.y0, detection.x1, detection.y1)
        draw.rectangle(coordinates, outline=(255, 48, 48), width=width)
        label = f"mine {detection.confidence:.3f}"
        left, top, right, bottom = draw.textbbox(
            (detection.x0, detection.y0), label, font=font
        )
        draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=(255, 48, 48))
        draw.text((detection.x0, detection.y0), label, fill=(255, 255, 255), font=font)
    return overlay


def predict_tiled(
    model,
    image,
    *,
    tile: int,
    overlap: int,
    batch: int,
    device: str | None = None,
) -> tuple[list[tuple[int, int]], list[Detection]]:
    """Return every class-0 deployment-tile candidate down to 0.001.

    Thresholding and cross-tile merging deliberately happen after this step so
    annotated evaluation can report both the candidate floor and the frozen
    operating threshold from the exact same inference pass.
    """
    origins = tile_grid(image.width, image.height, tile, overlap)
    crops = [image.crop((x, y, x + tile, y + tile)) for x, y in origins]
    candidates = []
    for (x, y), result in zip(
        origins,
        _predict_chunks(model, crops, batch, tile, device),
        strict=True,
    ):
        for xyxy, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            strict=True,
        ):
            confidence = float(confidence)
            if int(class_id) != 0 or confidence < CANDIDATE_FLOOR:
                continue
            x0, y0, x1, y1 = map(float, xyxy)
            candidates.append(
                Detection(
                    x0=max(0.0, min(image.width, x0 + x)),
                    y0=max(0.0, min(image.height, y0 + y)),
                    x1=max(0.0, min(image.width, x1 + x)),
                    y1=max(0.0, min(image.height, y1 + y)),
                    confidence=confidence,
                    tile_x=x,
                    tile_y=y,
                )
            )
    return origins, candidates


def run(
    weights: Path,
    inputs: Sequence[Path],
    out: Path,
    threshold: float,
    tile: int,
    overlap: int,
    batch: int,
    merge_overlap: float,
    device: str | None = None,
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    paths = _paths(inputs)
    if not paths:
        raise ValueError("no supported images found")
    out.mkdir(parents=True, exist_ok=True)
    overlays = out / "overlays"
    overlays.mkdir(exist_ok=True)

    from PIL import Image, ImageOps
    from ultralytics import YOLO

    model = YOLO(str(weights))
    records = []
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        origins, candidates = predict_tiled(
            model,
            image,
            tile=tile,
            overlap=overlap,
            batch=batch,
            device=device,
        )
        raw = [candidate for candidate in candidates if candidate.confidence >= threshold]
        merged = merge_detections(raw, merge_overlap)
        overlay_name = f"{index:03d}_{path.stem}_overlay.jpg"
        _draw_overlay(image, merged).save(overlays / overlay_name, quality=92)
        records.append(
            {
                "source": str(path),
                "source_sha256": sha256(path),
                "width": image.width,
                "height": image.height,
                "tiles": len(origins),
                "raw_tile_detections": len(raw),
                "detections": [asdict(detection) for detection in merged],
                "candidate_tile_detections": len(candidates),
                "candidates": [asdict(detection) for detection in candidates],
                "overlay": str(Path("overlays") / overlay_name),
            }
        )
        print(
            f"{path.name}: {len(merged)} merged detections "
            f"from {len(raw)} tile detections"
        )

    report = {
        "schema": SCHEMA,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "threshold": threshold,
        "candidate_floor": CANDIDATE_FLOOR,
        "tile_px": tile,
        "overlap_px": overlap,
        "merge_overlap": merge_overlap,
        "images": records,
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.37)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--merge-overlap", type=float, default=0.5)
    parser.add_argument(
        "--device",
        help="Ultralytics device (for example 0 or cpu); defaults to auto",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be in [0, 1]")
    if args.tile < 1 or not 0 <= args.overlap < args.tile:
        parser.error("need tile > 0 and 0 <= overlap < tile")
    if args.batch < 1:
        parser.error("batch must be positive")
    if not 0.0 <= args.merge_overlap <= 1.0:
        parser.error("merge overlap must be in [0, 1]")
    run(
        args.weights,
        args.images,
        args.out,
        args.threshold,
        args.tile,
        args.overlap,
        args.batch,
        args.merge_overlap,
        args.device,
    )


if __name__ == "__main__":
    main()
