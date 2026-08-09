"""Render deployment-style tiled YOLO detections onto a video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from audit_irl import merge_detections, predict_tiled


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def read_frame(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def draw_overlay(image, detections, *, frame: int, fps: float, threshold: float):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 23) if font_path.exists() else ImageFont.load_default()
    box_width = max(4, round(min(image.size) / 300))
    color = (255, 48, 48)

    for detection in detections:
        coordinates = (detection.x0, detection.y0, detection.x1, detection.y1)
        draw.rectangle(coordinates, outline=color, width=box_width)
        label = f"mine {detection.confidence:.3f}"
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        label_width = text_box[2] - text_box[0]
        label_height = text_box[3] - text_box[1]
        label_x = max(0, min(round(detection.x0), image.width - label_width - 10))
        label_y = max(0, round(detection.y0) - label_height - 10)
        draw.rectangle(
            (label_x, label_y, label_x + label_width + 10, label_y + label_height + 8),
            fill=color,
        )
        draw.text(
            (label_x + 5, label_y + 2),
            label,
            fill=(255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=color,
        )

    hud = (
        f"appearance-fold1  conf >= {threshold:.2f}  640/192 tiled  "
        f"t={frame / fps:06.1f}s  detections={len(detections)}"
    )
    hud_box = draw.textbbox((0, 0), hud, font=font)
    draw.rectangle(
        (10, 10, hud_box[2] - hud_box[0] + 28, hud_box[3] - hud_box[1] + 26),
        fill=(0, 0, 0),
    )
    draw.text((19, 16), hud, fill=(255, 255, 255), font=font)
    return image


def run(args: argparse.Namespace) -> None:
    from PIL import Image
    from ultralytics import YOLO

    source_info = probe(args.input)
    width = int(source_info["width"])
    height = int(source_info["height"])
    fps = float(Fraction(source_info["avg_frame_rate"]))
    frame_size = width * height * 3
    part = args.output.with_name(args.output.stem + ".part" + args.output.suffix)
    if args.output.exists() or part.exists() or args.report.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(args.input),
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
    if args.encoder == "libx264":
        encoder_options = [
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.quality),
        ]
    elif args.encoder == "h264_nvenc":
        encoder_options = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            args.preset,
            "-cq",
            str(args.quality),
            "-b:v",
            "0",
        ]
    else:
        raise ValueError(f"unsupported encoder: {args.encoder}")

    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            *encoder_options,
            "-pix_fmt",
            "yuv420p",
            "-metadata",
            f"comment=Mission 10 tiled YOLO overlay; threshold={args.threshold}",
            str(part),
        ],
        stdin=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    assert encoder.stdin is not None

    model = YOLO(str(args.weights))
    records = []
    started = time.monotonic()
    frame_index = 0
    try:
        while True:
            raw_frame = read_frame(decoder.stdout, frame_size)
            if not raw_frame:
                break
            if len(raw_frame) != frame_size:
                raise RuntimeError(
                    f"truncated decoded frame {frame_index}: {len(raw_frame)}/{frame_size} bytes"
                )
            image = Image.frombytes("RGB", (width, height), raw_frame)
            _, candidates = predict_tiled(
                model,
                image,
                tile=args.tile,
                overlap=args.overlap,
                batch=args.batch,
                device=args.device,
            )
            selected = [
                detection
                for detection in candidates
                if detection.confidence >= args.threshold
            ]
            detections = merge_detections(selected, args.merge_overlap)
            draw_overlay(
                image,
                detections,
                frame=frame_index,
                fps=fps,
                threshold=args.threshold,
            )
            encoder.stdin.write(image.tobytes())
            records.append(
                {
                    "frame": frame_index,
                    "time_seconds": frame_index / fps,
                    "detections": [asdict(detection) for detection in detections],
                }
            )
            frame_index += 1
            if frame_index % 50 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"processed {frame_index} frames ({frame_index / elapsed:.2f} fps)",
                    flush=True,
                )
    finally:
        decoder.stdout.close()
        encoder.stdin.close()

    decoder_status = decoder.wait()
    encoder_status = encoder.wait()
    if decoder_status or encoder_status:
        raise RuntimeError(
            f"ffmpeg failed: decoder={decoder_status}, encoder={encoder_status}"
        )
    part.replace(args.output)

    output_info = probe(args.output)
    report = {
        "schema": "mission10-yolo-video-overlay/1",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "weights": str(args.weights),
        "weights_sha256": sha256(args.weights),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_video": source_info,
        "output_video": output_info,
        "threshold": args.threshold,
        "tile": args.tile,
        "overlap": args.overlap,
        "merge_overlap": args.merge_overlap,
        "frames_processed": frame_index,
        "frames_with_detections": sum(bool(record["detections"]) for record in records),
        "detections_total": sum(len(record["detections"]) for record in records),
        "elapsed_seconds": time.monotonic() - started,
        "frames": records,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "frames_processed", "frames_with_detections", "detections_total", "elapsed_seconds"
    )}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--merge-overlap", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="libx264")
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--quality", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
