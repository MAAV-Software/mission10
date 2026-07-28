"""Shared I/O and numerical helpers for the localization replay experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def split_mcaps(bag: Path) -> list[Path]:
    if bag.is_file():
        return [bag]
    files = list(bag.glob("*.mcap"))
    if not files:
        raise FileNotFoundError(f"no MCAP files under {bag}")

    def key(path: Path) -> tuple[str, int]:
        stem, _, suffix = path.stem.rpartition("_")
        return stem, int(suffix) if suffix.isdigit() else 0

    return sorted(files, key=key)


def iter_messages(
    bag: Path, topics: Iterable[str] | None = None
) -> Iterator[tuple[str, int, object]]:
    # Keep MCAP optional for runners that only consume an already prepared
    # image/CSV dataset, such as the self-contained SVO frontend benchmark.
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    selected = list(topics) if topics is not None else None
    for path in split_mcaps(bag):
        with path.open("rb") as stream:
            reader = make_reader(
                stream, decoder_factories=[DecoderFactory()]
            )
            for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=selected
            ):
                yield channel.topic, int(message.log_time), decoded


def robust_affine(x, y) -> tuple[float, float, float]:
    """Fit y=a*x+b and trim one three-sigma residual pass."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        raise ValueError(f"cannot fit clocks with x={len(x)}, y={len(y)}")
    x0 = float(x.mean())
    a, centered_b = np.polyfit(x - x0, y, 1)
    residual = y - (a * (x - x0) + centered_b)
    sigma = float(residual.std())
    if sigma > 0:
        keep = np.abs(residual) <= 3.0 * sigma
        if 2 < int(keep.sum()) < len(x):
            a, centered_b = np.polyfit(
                x[keep] - x0, y[keep], 1
            )
            residual = y[keep] - (
                a * (x[keep] - x0) + centered_b
            )
            sigma = float(residual.std())
    return float(a), float(centered_b - a * x0), sigma


def write_pgm(path: Path, image: np.ndarray) -> None:
    height, width = image.shape
    with path.open("wb") as stream:
        stream.write(f"P5\n{width} {height}\n255\n".encode())
        stream.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())


def image_to_gray(message) -> np.ndarray:
    height, width = int(message.height), int(message.width)
    step = int(message.step)
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    encoding = str(message.encoding).lower()
    rows = data.reshape(height, step)
    if encoding in {"mono8", "8uc1"}:
        return rows[:, :width].copy()
    if encoding in {"yuyv422", "yuyv", "yuv422_yuy2"}:
        return rows[:, : 2 * width : 2].copy()
    raise ValueError(
        f"unsupported image encoding {message.encoding!r} "
        f"for {width}x{height}, step={step}"
    )


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentile(values, q: float):
    values = [float(value) for value in values if finite(value)]
    return float(np.percentile(values, q)) if values else None


def median(values):
    values = [float(value) for value in values if finite(value)]
    return float(np.median(values)) if values else None
