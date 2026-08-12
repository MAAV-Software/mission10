"""Retain tiled YOLO candidates for every frame of a native CM2 video."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from audit_irl import CANDIDATE_FLOOR, predict_tiled


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if args.output.exists() or args.meta.exists():
        raise FileExistsError("refusing to overwrite candidate output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from ultralytics import YOLO

    video = probe(args.input)
    width = int(video["width"])
    height = int(video["height"])
    fps = float(Fraction(video["avg_frame_rate"]))
    frame_bytes = width * height * 3
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
    assert decoder.stdout is not None
    model = YOLO(str(args.weights))
    started = time.monotonic()
    frame_index = 0
    candidate_count = 0
    frames_with_candidates = 0

    with gzip.open(args.output, "wt", encoding="utf-8") as output:
        while True:
            pixels = read_frame(decoder.stdout, frame_bytes)
            if not pixels:
                break
            if len(pixels) != frame_bytes:
                raise RuntimeError(
                    f"truncated frame {frame_index}: {len(pixels)}/{frame_bytes}"
                )
            image = Image.frombytes("RGB", (width, height), pixels)
            origins, candidates = predict_tiled(
                model,
                image,
                tile=args.tile,
                overlap=args.overlap,
                batch=args.batch,
                device=args.device,
            )
            output.write(
                json.dumps(
                    {
                        "frame": frame_index,
                        "video_time_s": frame_index / fps,
                        "tiles": len(origins),
                        "candidates": [asdict(candidate) for candidate in candidates],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            candidate_count += len(candidates)
            frames_with_candidates += bool(candidates)
            frame_index += 1
            if frame_index % 100 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"frames={frame_index} rate={frame_index / elapsed:.2f} fps "
                    f"candidates={candidate_count}",
                    flush=True,
                )

    decoder.stdout.close()
    status = decoder.wait()
    if status:
        raise RuntimeError(f"ffmpeg decoder failed with status {status}")
    if args.expected_frames is not None and frame_index != args.expected_frames:
        raise RuntimeError(
            f"frame count mismatch: decoded {frame_index}, expected {args.expected_frames}"
        )

    report = {
        "schema": "mission10-yolo-video-candidates/1",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "video": video,
        "frames": frame_index,
        "frames_with_candidates": frames_with_candidates,
        "candidates": candidate_count,
        "candidate_floor": CANDIDATE_FLOOR,
        "tile": args.tile,
        "overlap": args.overlap,
        "batch": args.batch,
        "device": args.device,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.meta.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
