#!/usr/bin/env python3
"""Extract sparse, lossless CM2 frames from a split ROS 2 MCAP bag."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw


def split_paths(bag: Path) -> list[Path]:
    def split_number(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[1])

    return sorted(bag.glob("*.mcap"), key=split_number)


def stamp_ns(header: object) -> int:
    stamp = header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def first_stamp(paths: list[Path], topic: str) -> int:
    for path in paths:
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=[topic]):
                return stamp_ns(msg.header)
    raise RuntimeError(f"topic has no messages: {topic}")


def image_from_message(msg: object) -> Image.Image:
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    encoding = str(msg.encoding).lower()
    pixels = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
    if encoding == "mono8":
        return Image.fromarray(pixels[:, :width].copy(), mode="L")
    if encoding in {"rgb8", "bgr8"}:
        image = pixels[:, : width * 3].reshape(height, width, 3).copy()
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return Image.fromarray(image, mode="RGB")
    if encoding in {"yuyv", "yuy2", "yuv422_yuy2"}:
        if width % 2:
            raise RuntimeError(f"YUYV image width must be even: {width}")
        packed = pixels[:, : width * 2].reshape(height, width // 2, 4)
        luminance = np.empty((height, width), dtype=np.int32)
        chroma_u = np.empty((height, width), dtype=np.int32)
        chroma_v = np.empty((height, width), dtype=np.int32)
        luminance[:, 0::2] = packed[:, :, 0]
        luminance[:, 1::2] = packed[:, :, 2]
        chroma_u[:, 0::2] = packed[:, :, 1]
        chroma_u[:, 1::2] = packed[:, :, 1]
        chroma_v[:, 0::2] = packed[:, :, 3]
        chroma_v[:, 1::2] = packed[:, :, 3]

        # V4L2 YUYV uses the BT.601 limited-range conversion. Integer math
        # mirrors the conventional decoder and keeps extraction independent of
        # OpenCV, which is otherwise unnecessary for this review utility.
        c = luminance - 16
        d = chroma_u - 128
        e = chroma_v - 128
        rgb = np.stack(
            (
                (298 * c + 409 * e + 128) >> 8,
                (298 * c - 100 * d - 208 * e + 128) >> 8,
                (298 * c + 516 * d + 128) >> 8,
            ),
            axis=-1,
        )
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    raise RuntimeError(
        f"unsupported image format: {msg.width}x{msg.height} {msg.encoding}"
    )


def write_contact_sheets(records: list[dict[str, object]], output: Path) -> None:
    columns = 4
    rows = 3
    thumb_size = (410, 308)
    label_height = 24
    per_sheet = columns * rows
    for sheet_index in range(0, len(records), per_sheet):
        subset = records[sheet_index : sheet_index + per_sheet]
        mode = "L" if all(record["encoding"] == "mono8" for record in subset) else "RGB"
        sheet = Image.new(
            mode,
            (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
            0,
        )
        draw = ImageDraw.Draw(sheet)
        ink = 255 if mode == "L" else (255, 255, 255)
        for offset, record in enumerate(subset):
            frame = Image.open(output / str(record["file"]))
            frame.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            x = (offset % columns) * thumb_size[0]
            y = (offset // columns) * (thumb_size[1] + label_height)
            sheet.paste(frame, (x, y))
            draw.text((x + 5, y + thumb_size[1] + 4), f"t={record['requested_t_s']:.1f}s", fill=ink)
        number = sheet_index // per_sheet + 1
        sheet.save(output / f"contact_sheet_{number:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--topic", default="/camera_down/image_raw")
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--step-s", type=float, default=2.0)
    args = parser.parse_args()

    if args.step_s <= 0 or args.end_s < args.start_s:
        parser.error("require --step-s > 0 and --end-s >= --start-s")

    paths = split_paths(args.bag)
    if not paths:
        raise RuntimeError(f"no split MCAP files found in {args.bag}")
    origin_ns = first_stamp(paths, args.topic)
    requested_seconds = np.arange(
        args.start_s, args.end_s + args.step_s * 0.5, args.step_s
    ).tolist()
    targets = [origin_ns + round(seconds * 1e9) for seconds in requested_seconds]
    lower_ns = targets[0] - 500_000_000
    upper_ns = targets[-1] + 500_000_000

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    target_index = 0
    for path in paths:
        if target_index == len(targets):
            break
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            messages = reader.iter_decoded_messages(
                topics=[args.topic], start_time=lower_ns, end_time=upper_ns
            )
            for _, _, message, msg in messages:
                timestamp_ns = stamp_ns(msg.header)
                while target_index < len(targets) and timestamp_ns >= targets[target_index]:
                    frame = image_from_message(msg)
                    requested_t_s = requested_seconds[target_index]
                    name = f"cm2_t{requested_t_s:07.1f}s.png"
                    destination = args.output / name
                    frame.save(destination, compress_level=3)
                    records.append(
                        {
                            "file": name,
                            "requested_t_s": requested_t_s,
                            "actual_t_s": (timestamp_ns - origin_ns) * 1e-9,
                            "header_timestamp_ns": timestamp_ns,
                            "log_time_ns": int(message.log_time),
                            "source_split": path.name,
                            "encoding": str(msg.encoding).lower(),
                            "width": int(msg.width),
                            "height": int(msg.height),
                            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        }
                    )
                    target_index += 1
                if target_index == len(targets):
                    break

    if not records:
        raise RuntimeError("no requested frames were available")
    if target_index != len(targets):
        print(
            f"warning: topic ended after {target_index}/{len(targets)} requested frames",
            file=sys.stderr,
        )

    manifest = {
        "schema_version": 1,
        "bag": str(args.bag.resolve()),
        "topic": args.topic,
        "origin_header_timestamp_ns": origin_ns,
        "sampling": {
            "start_s": args.start_s,
            "end_s": args.end_s,
            "step_s": args.step_s,
            "selection": "first image whose header timestamp is at or after each target",
        },
        "frames": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_contact_sheets(records, args.output)
    print(f"extracted {len(records)} frames to {args.output}")


if __name__ == "__main__":
    main()
