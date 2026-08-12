"""Render retained CM2 candidates as a size-bounded two-pass VP9 MP4."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

from audit_irl import Detection, merge_detections


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream=codec_name,codec_tag_string,width,height,avg_frame_rate,"
                "nb_frames,nb_read_frames,duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def read_frame(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


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


def finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_telemetry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for frame, row in enumerate(rows):
        if int(row["frame"]) != frame:
            raise ValueError(f"non-contiguous telemetry frame {frame}: {path}")
    return rows


def candidate_record(stream, expected_frame: int) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise ValueError(f"candidate report ended before frame {expected_frame}")
    record = json.loads(line)
    if int(record["frame"]) != expected_frame:
        raise ValueError(
            f"candidate frame mismatch: {record['frame']} != {expected_frame}"
        )
    return record


def selected_detections(record: dict[str, Any], threshold: float) -> list[Detection]:
    selected = [
        Detection(**item)
        for item in record["candidates"]
        if float(item["confidence"]) >= threshold
    ]
    return merge_detections(selected, 0.5)


def draw_overlay(
    image,
    detections: list[Detection],
    telemetry: dict[str, str],
    *,
    bag: str,
    frame: int,
    fps: float,
    threshold: float,
) -> None:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = (
        ImageFont.truetype(str(font_path), 23)
        if font_path.exists()
        else ImageFont.load_default()
    )
    color = (255, 48, 48)
    for detection in detections:
        box = (detection.x0, detection.y0, detection.x1, detection.y1)
        draw.rectangle(box, outline=color, width=5)
        label = f"mine {detection.confidence:.3f}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_x = max(0, min(round(detection.x0), image.width - label_width - 10))
        label_y = max(0, round(detection.y0) - label_height - 10)
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + label_width + 10,
                label_y + label_height + 8,
            ),
            fill=color,
        )
        draw.text((label_x + 5, label_y + 2), label, fill="white", font=font)

    range_m = finite(telemetry.get("range_current_distance", ""))
    vx = finite(telemetry.get("local_vx", ""))
    vy = finite(telemetry.get("local_vy", ""))
    speed = math.hypot(vx, vy) if vx is not None and vy is not None else None
    state = "OPERATIONAL" if operational(telemetry) else "NON-OPERATIONAL"
    range_text = f"{range_m:.2f}m" if range_m is not None else "n/a"
    speed_text = f"{speed:.2f}m/s" if speed is not None else "n/a"
    hud = (
        f"{bag}  appearance-fold1 >= {threshold:.2f}  "
        f"t={frame / fps:06.1f}s  range={range_text}  speed={speed_text}  "
        f"{state}  detections={len(detections)}"
    )
    hud_box = draw.textbbox((0, 0), hud, font=font)
    draw.rectangle((10, 10, hud_box[2] + 28, hud_box[3] + 26), fill="black")
    draw.text((19, 16), hud, fill="white", font=font)


def encoder_command(
    args: argparse.Namespace,
    *,
    width: int,
    height: int,
    fps: float,
    bitrate: int,
    pass_number: int,
    part: Path,
) -> list[str]:
    common = [
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
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        str(bitrate),
        "-deadline",
        "good",
        "-cpu-used",
        str(args.first_pass_cpu_used if pass_number == 1 else args.cpu_used),
        "-row-mt",
        "1",
        "-threads",
        str(args.threads),
        "-tile-columns",
        "2",
        "-frame-parallel",
        "1",
        "-pass",
        str(pass_number),
        "-passlogfile",
        str(args.passlog),
    ]
    if pass_number == 1:
        return [*common, "-f", "null", "/dev/null"]
    return [
        *common,
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "vp09",
        "-movflags",
        "+faststart",
        "-metadata",
        f"comment=Mission 10 appearance-fold1; threshold={args.threshold}",
        str(part),
    ]


def render_pass(
    args: argparse.Namespace,
    telemetry: list[dict[str, str]],
    *,
    width: int,
    height: int,
    fps: float,
    bitrate: int,
    pass_number: int,
) -> dict[str, int]:
    from PIL import Image

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
    part = args.output.with_name(args.output.stem + ".part" + args.output.suffix)
    encoder = subprocess.Popen(
        encoder_command(
            args,
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            pass_number=pass_number,
            part=part,
        ),
        stdin=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    assert encoder.stdin is not None
    frame_size = width * height * 3
    frame = 0
    frames_with_detections = 0
    detections_total = 0
    started = time.monotonic()
    with gzip.open(args.candidates, "rt", encoding="utf-8") as candidates:
        try:
            while frame < len(telemetry):
                pixels = read_frame(decoder.stdout, frame_size)
                if not pixels:
                    break
                if len(pixels) != frame_size:
                    raise RuntimeError(f"truncated decoded frame {frame}")
                record = candidate_record(candidates, frame)
                detections = selected_detections(record, args.threshold)
                image = Image.frombytes("RGB", (width, height), pixels)
                draw_overlay(
                    image,
                    detections,
                    telemetry[frame],
                    bag=args.bag,
                    frame=frame,
                    fps=fps,
                    threshold=args.threshold,
                )
                encoder.stdin.write(image.tobytes())
                frames_with_detections += bool(detections)
                detections_total += len(detections)
                frame += 1
                if frame % 100 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"pass={pass_number} frames={frame}/{len(telemetry)} "
                        f"rate={frame / elapsed:.2f} fps",
                        flush=True,
                    )
            if frame != len(telemetry):
                raise ValueError(f"decoded {frame} frames; expected {len(telemetry)}")
            if candidates.readline():
                raise ValueError("candidate report has frames beyond telemetry")
        finally:
            decoder.stdout.close()
            encoder.stdin.close()
    decoder_status = decoder.wait()
    encoder_status = encoder.wait()
    if decoder_status or encoder_status:
        raise RuntimeError(
            f"ffmpeg failed: decoder={decoder_status}, encoder={encoder_status}"
        )
    return {
        "frames": frame,
        "frames_with_detections": frames_with_detections,
        "detections": detections_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--target-mb", type=float, default=18.0)
    parser.add_argument("--maximum-mb", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-used", type=int, default=4)
    parser.add_argument("--first-pass-cpu-used", type=int, default=6)
    parser.add_argument("--passlog", type=Path, required=True)
    args = parser.parse_args()

    part = args.output.with_name(args.output.stem + ".part" + args.output.suffix)
    occupied = (args.output, part, args.report)
    if any(path.exists() for path in occupied):
        raise FileExistsError(f"refusing to overwrite output: {occupied}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.passlog.parent.mkdir(parents=True, exist_ok=True)

    source = probe(args.input)
    width = int(source["width"])
    height = int(source["height"])
    fps = float(Fraction(source["avg_frame_rate"]))
    telemetry = load_telemetry(args.frames)
    duration = len(telemetry) / fps
    bitrate = round(args.target_mb * 1_000_000 * 8 * 0.985 / duration)

    meta = json.loads(args.candidate_meta.read_text())
    if sha256(args.candidates) != meta["output_sha256"]:
        raise ValueError("candidate report SHA-256 does not match its metadata")
    if int(meta["frames"]) != len(telemetry):
        raise ValueError("candidate and telemetry frame counts differ")
    if meta["input_sha256"] != sha256(args.input):
        raise ValueError("candidate report and source video SHA-256 differ")

    started = time.monotonic()
    first = render_pass(
        args,
        telemetry,
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        pass_number=1,
    )
    second = render_pass(
        args,
        telemetry,
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        pass_number=2,
    )
    if first != second:
        raise ValueError("render passes produced different detection summaries")
    part.replace(args.output)
    size = args.output.stat().st_size
    if size > args.maximum_mb * 1_000_000:
        raise ValueError(
            f"output is {size / 1_000_000:.2f} MB; maximum is {args.maximum_mb:.2f} MB"
        )
    output_probe = probe(args.output)
    output_frames = int(output_probe.get("nb_read_frames") or 0)
    if output_frames != len(telemetry):
        raise ValueError(f"output has {output_frames}/{len(telemetry)} frames")
    if output_probe.get("codec_name") != "vp9":
        raise ValueError(f"output codec is {output_probe.get('codec_name')}, not VP9")

    report = {
        "schema": "mission10-cm2-candidate-overlay-video/1",
        "bag": args.bag,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "frames_csv": str(args.frames.resolve()),
        "frames_csv_sha256": sha256(args.frames),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": sha256(args.candidates),
        "weights_sha256": meta["weights_sha256"],
        "threshold": args.threshold,
        "merge_overlap": 0.5,
        "target_mb": args.target_mb,
        "maximum_mb": args.maximum_mb,
        "bitrate_bps": bitrate,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_bytes": size,
        "output_video": output_probe,
        "summary": second,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
