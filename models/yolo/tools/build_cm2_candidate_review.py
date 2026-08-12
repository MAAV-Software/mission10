"""Build a compact paired-model review set from CM2 candidate reports."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BAG_DIRECTORIES = {
    "manual": "manual_survey",
    "return": "return_failure",
    "petal": "petal_qual",
    "survey": "survey",
}
DEVELOPMENT_BAGS = ("manual", "return", "petal")
COLORS = {"appearance": (255, 48, 48), "production": (48, 200, 255)}


def truth(value: str) -> bool:
    return value.lower() == "true"


def operational(row: dict[str, str]) -> bool:
    try:
        return (
            truth(row["flags_cs_in_air"])
            and int(row["status_arming_state"]) == 2
            and float(row["range_current_distance"]) >= 0.4
        )
    except (KeyError, TypeError, ValueError):
        return False


def overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
    x0 = max(first["x0"], second["x0"])
    y0 = max(first["y0"], second["y0"])
    x1 = min(first["x1"], second["x1"])
    y1 = min(first["y1"], second["y1"])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_first = max(0.0, first["x1"] - first["x0"]) * max(
        0.0, first["y1"] - first["y0"]
    )
    area_second = max(0.0, second["x1"] - second["x0"]) * max(
        0.0, second["y1"] - second["y0"]
    )
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def select_frames(
    bag: str,
    frames: list[dict[str, str]],
    comparison: dict[str, Any],
    sources: dict[int, dict[str, Any]],
    threshold: float,
    maximum_frames: int,
) -> list[dict[str, Any]]:
    selected: dict[int, set[str]] = {}

    def add(frame: int, reason: str, *, minimum_gap: int = 5) -> bool:
        if frame in selected:
            selected[frame].add(reason)
            return True
        if any(abs(frame - existing) < minimum_gap for existing in selected):
            return False
        selected[frame] = {reason}
        return True

    bag_report = comparison["bags"][bag]["models"]
    threshold_key = f"{threshold:.2f}"
    appearance_tracks = bag_report["appearance"][threshold_key]["persistent_track_records"]
    production_tracks = bag_report["production"][threshold_key]["persistent_track_records"]
    for track in sorted(appearance_tracks, key=lambda item: item["peak_confidence"], reverse=True):
        if sum("appearance-track" in reasons for reasons in selected.values()) >= 12:
            break
        add(track["peak_frame"], "appearance-track")
    for track in sorted(production_tracks, key=lambda item: item["peak_confidence"], reverse=True):
        if sum("production-track" in reasons for reasons in selected.values()) >= 8:
            break
        add(track["peak_frame"], "production-track")

    disagreements = []
    for frame, models in sources.items():
        appearance = [
            item["box"] for item in models.get(f"appearance_{threshold_key}", [])
        ]
        production = [
            item["box"] for item in models.get(f"production_{threshold_key}", [])
        ]
        unmatched = [
            box
            for box in appearance
            if all(overlap(box, other) < 0.5 for other in production)
        ] + [
            box
            for box in production
            if all(overlap(box, other) < 0.5 for other in appearance)
        ]
        if unmatched:
            disagreements.append((max(box["confidence"] for box in unmatched), frame))
    for _, frame in sorted(disagreements, reverse=True):
        if sum("model-disagreement" in reasons for reasons in selected.values()) >= 6:
            break
        add(frame, "model-disagreement")

    operational_frames = [index for index, row in enumerate(frames) if operational(row)]
    if operational_frames:
        for offset in range(8):
            index = round(offset * (len(operational_frames) - 1) / 7) if offset else 0
            add(operational_frames[index], "systematic", minimum_gap=0)

    ranked = []
    for frame, models in sources.items():
        confidence = max(
            (
                item["box"]["confidence"]
                for items in models.values()
                for item in items
            ),
            default=0.0,
        )
        ranked.append((confidence, frame))
    for _, frame in sorted(ranked, reverse=True):
        if len(selected) >= maximum_frames:
            break
        add(frame, "high-confidence", minimum_gap=3)
    for frame in operational_frames:
        if len(selected) >= maximum_frames:
            break
        add(frame, "systematic-fill", minimum_gap=0)

    return [
        {
            "bag": bag,
            "frame": frame,
            "camera_time_s": float(frames[frame]["camera_time_s"]),
            "range_m": float(frames[frame]["range_current_distance"]),
            "speed_mps": math_hypot(frames[frame]["local_vx"], frames[frame]["local_vy"]),
            "reasons": sorted(reasons),
            "detections": sources.get(frame, {}),
        }
        for frame, reasons in sorted(selected.items())
    ]


def math_hypot(first: str, second: str) -> float:
    return (float(first) ** 2 + float(second) ** 2) ** 0.5


def draw_overlay(
    image: Image.Image, record: dict[str, Any], threshold: float
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    for model in ("production", "appearance"):
        for item in record["detections"].get(f"{model}_{threshold:.2f}", []):
            box = item["box"]
            color = COLORS[model]
            draw.rectangle((box["x0"], box["y0"], box["x1"], box["y1"]), outline=color, width=5)
            label = f"{model[0].upper()} {box['confidence']:.3f}"
            draw.text((box["x0"] + 3, max(3, box["y0"] - 27)), label, fill=color, font=font)
    hud = (
        f"{record['bag']} frame={record['frame']} t={record['camera_time_s']:.1f}s "
        f"range={record['range_m']:.2f}m speed={record['speed_mps']:.2f}m/s  "
        "A=red P=cyan"
    )
    box = draw.textbbox((0, 0), hud, font=font)
    draw.rectangle((8, 8, box[2] + 24, box[3] + 22), fill=(0, 0, 0))
    draw.text((16, 12), hud, fill=(255, 255, 255), font=font)
    return image


def read_frame(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def extract_video(
    video: Path,
    records: list[dict[str, Any]],
    output: Path,
    threshold: float,
) -> None:
    selected = {record["frame"]: record for record in records}
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    frame_size = 1640 * 1232 * 3
    index = 0
    while True:
        pixels = read_frame(decoder.stdout, frame_size)
        if not pixels:
            break
        if len(pixels) != frame_size:
            raise RuntimeError(f"truncated decoded frame {index}")
        if index in selected:
            image = Image.frombytes("RGB", (1640, 1232), pixels)
            image.save(output / "raw" / f"{selected[index]['bag']}_f{index:04d}.png", compress_level=3)
            draw_overlay(image, selected[index], threshold).save(
                output / "overlays" / f"{selected[index]['bag']}_f{index:04d}_overlay.jpg",
                quality=94,
            )
        index += 1
    decoder.stdout.close()
    if decoder.wait():
        raise RuntimeError(f"ffmpeg failed while decoding {video}")


def contact_sheets(records: list[dict[str, Any]], output: Path) -> None:
    columns, rows = 4, 3
    thumb = (410, 308)
    label_height = 26
    per_sheet = columns * rows
    for start in range(0, len(records), per_sheet):
        subset = records[start : start + per_sheet]
        sheet = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + label_height)))
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(subset):
            path = output / "overlays" / f"{record['bag']}_f{record['frame']:04d}_overlay.jpg"
            with Image.open(path) as source:
                image = source.copy()
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = (offset % columns) * thumb[0]
            y = (offset // columns) * (thumb[1] + label_height)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb[1] + 4), f"{record['bag']} f{record['frame']}", fill="white")
        number = start // per_sheet + 1
        sheet.save(output / f"contact_sheet_{number:02d}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--comparison-directory", default="development-comparison")
    parser.add_argument(
        "--bags",
        nargs="+",
        choices=tuple(BAG_DIRECTORIES),
        default=DEVELOPMENT_BAGS,
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--maximum-frames-per-bag", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output}")
    (args.output / "raw").mkdir(parents=True, exist_ok=True)
    (args.output / "overlays").mkdir(exist_ok=True)

    comparison = json.loads(
        (args.archive / args.comparison_directory / "comparison.json").read_text()
    )
    review_document = json.loads(
        (args.archive / args.comparison_directory / "review_sources.json").read_text()
    )
    all_records = []
    for bag in args.bags:
        directory = BAG_DIRECTORIES[bag]
        with (args.archive / "bags" / directory / "frames.csv").open(newline="") as stream:
            frames = list(csv.DictReader(stream))
        sources = {record["frame"]: record["detections"] for record in review_document[bag]}
        records = select_frames(
            bag,
            frames,
            comparison,
            sources,
            args.threshold,
            args.maximum_frames_per_bag,
        )
        all_records.extend(records)
        extract_video(
            args.archive / "bags" / directory / "cm2_native_crf12.mkv",
            records,
            args.output,
            args.threshold,
        )
    (args.output / "selection.json").write_text(json.dumps(all_records, indent=2) + "\n")
    contact_sheets(all_records, args.output)
    print(f"selected {len(all_records)} frames into {args.output}")


if __name__ == "__main__":
    main()
