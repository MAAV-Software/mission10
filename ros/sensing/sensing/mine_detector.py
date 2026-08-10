"""Hailo mine detection as an external CM2 pool consumer.

This module is the rendezvous point between mission sensing and the mission
runner's mine bookkeeping.  The sensing process owns the camera; this consumer
runs inside the mission process, attaches to the ``cm2`` shared-memory pool
read-only, copies the freshest frame at a fixed cadence, tiles it for the
640 px detector, and puts one item per detected box on a caller-owned
``queue.Queue``:

    (realtime_ns, x, y, w, h)   # top-left pixel coordinates in the full
                                # 1640 x 1232 frame

``realtime_ns`` is the frame's realtime stamp; every box from one frame
carries the same stamp.  A frame with no detections queues nothing.  Use a
bounded queue: when it is full the oldest item is dropped, so a stalled
consumer costs boxes, never detection cadence.

Two inference backends share one output convention, a list of
``(x, y, w, h, score)`` tuples in tile pixels:

- ``HailoYoloBackend``: the flight path.  Runs the compiled single-class
  YOLOv11m HEF on the Hailo-8 through HailoRT; NMS is embedded in the HEF.
- ``OnnxYoloBackend``: bench fallback for machines without a Hailo.  Decodes
  the raw Ultralytics ONNX head and applies NMS here.

Neither backend is imported unless constructed, so this module stays
importable on machines with neither runtime.
"""
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .shared_frame_pool import FrameMetadata, SharedFramePool


TILE = 640
DEFAULT_CONFIDENCE = 0.3
DEFAULT_DEDUP_IOU = 0.45
# Zero means device-bound: poll the newest frame as soon as the previous
# detection pass finishes (about 7-8 fps with sequential batch-1 tiles).
DEFAULT_PERIOD_S = 0.0
STALE_POOL_S = 5.0

DEFAULT_HEF = Path(
    os.environ.get(
        "MAAV_MINE_HEF",
        Path(__file__).resolve().parents[3]
        / "models/yolo/dataset/runpod-a4500-pilot40/artifacts"
        / "pilot40-yolo11m-640-hailo8.hef",
    )
)


def tile_origins(width: int, height: int, tile: int = TILE) -> list[tuple[int, int]]:
    """Evenly spaced tile origins that cover the frame with overlap.

    1640 x 1232 yields the six tiles the sensing design assumes
    (rfd-single-camera-sensing 3.1).
    """

    def axis(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        steps = math.ceil((extent - tile) / tile)
        stride = (extent - tile) / steps
        return [round(index * stride) for index in range(steps + 1)]

    return [(x, y) for y in axis(height) for x in axis(width)]


def yuyv_to_rgb(payload: bytes, width: int, height: int) -> np.ndarray:
    yuyv = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 2)
    try:
        import cv2

        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
    except ImportError:
        pass
    # Vectorized BT.601 limited-range fallback for hosts without OpenCV.
    luma = yuyv[:, :, 0].astype(np.float32) - 16.0
    chroma_u = np.repeat(yuyv[:, 0::2, 1], 2, axis=1)[:, :width].astype(np.float32) - 128.0
    chroma_v = np.repeat(yuyv[:, 1::2, 1], 2, axis=1)[:, :width].astype(np.float32) - 128.0
    red = 1.164 * luma + 1.596 * chroma_v
    green = 1.164 * luma - 0.392 * chroma_u - 0.813 * chroma_v
    blue = 1.164 * luma + 2.017 * chroma_u
    return np.clip(np.stack((red, green, blue), axis=-1), 0.0, 255.0).astype(np.uint8)


def dedup_boxes(
    boxes: Sequence[tuple[float, float, float, float, float]],
    iou_threshold: float = DEFAULT_DEDUP_IOU,
) -> list[tuple[float, float, float, float, float]]:
    """Greedy IoU suppression; overlapping tiles duplicate seam detections."""
    ordered = sorted(boxes, key=lambda box: box[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    for candidate in ordered:
        if all(_iou(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    return overlap / (aw * ah + bw * bh - overlap)


class OnnxYoloBackend:
    """Raw Ultralytics YOLOv11 ONNX head + host-side NMS, for benches."""

    def __init__(self, model_path, confidence: float = DEFAULT_CONFIDENCE) -> None:
        import onnxruntime as ort

        self.confidence = confidence
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, tile_rgb: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        image = tile_rgb.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))[np.newaxis]
        predictions = self.session.run(None, {self.input_name: image})[0][0].T
        boxes = []
        for center_x, center_y, width, height, score in predictions:
            if score >= self.confidence:
                boxes.append(
                    (
                        float(center_x - width / 2),
                        float(center_y - height / 2),
                        float(width),
                        float(height),
                        float(score),
                    )
                )
        return dedup_boxes(boxes)


class HailoYoloBackend:
    """Single-class YOLOv11 HEF with embedded NMS, through HailoRT."""

    def __init__(self, hef_path=DEFAULT_HEF, confidence: float = DEFAULT_CONFIDENCE) -> None:
        from hailo_platform import VDevice

        self.confidence = confidence
        self.tile = TILE
        params = VDevice.create_params()
        self.vdevice = VDevice(params)
        self.infer_model = self.vdevice.create_infer_model(str(hef_path))
        self.infer_model.set_batch_size(1)
        self.configured = self.infer_model.configure()
        self.configured.__enter__()

    def infer(self, tile_rgb: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        bindings = self.configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(tile_rgb))
        self.configured.run([bindings], timeout=2000)
        return self._parse_nms(bindings.output().get_buffer())

    def _parse_nms(self, buffer) -> list[tuple[float, float, float, float, float]]:
        """Decode HailoRT NMS-by-class output into tile-pixel boxes.

        Per class: one leading count, then count * [y_min, x_min, y_max,
        x_max, score], all normalized to [0, 1].  The pilot HEF has one class.
        """
        flat = np.asarray(buffer, dtype=np.float32).reshape(-1)
        boxes = []
        cursor = 0
        while cursor < flat.size:
            count = int(flat[cursor])
            cursor += 1
            for _ in range(count):
                y_min, x_min, y_max, x_max, score = flat[cursor : cursor + 5]
                cursor += 5
                if score >= self.confidence:
                    boxes.append(
                        (
                            float(x_min * self.tile),
                            float(y_min * self.tile),
                            float((x_max - x_min) * self.tile),
                            float((y_max - y_min) * self.tile),
                            float(score),
                        )
                    )
        return boxes

    def close(self) -> None:
        self.configured.__exit__(None, None, None)
        self.vdevice.release()


def detect_frame(
    backend,
    frame_rgb: np.ndarray,
    tile: int = TILE,
    iou_threshold: float = DEFAULT_DEDUP_IOU,
) -> list[tuple[float, float, float, float, float]]:
    """Tile one full frame, run the backend per tile, dedup seam duplicates."""
    height, width = frame_rgb.shape[:2]
    boxes = []
    for x0, y0 in tile_origins(width, height, tile):
        tile_view = frame_rgb[y0 : y0 + tile, x0 : x0 + tile]
        if tile_view.shape[0] < tile or tile_view.shape[1] < tile:
            padded = np.zeros((tile, tile, 3), dtype=frame_rgb.dtype)
            padded[: tile_view.shape[0], : tile_view.shape[1]] = tile_view
            tile_view = padded
        for x, y, w, h, score in backend.infer(tile_view):
            boxes.append((x + x0, y + y0, w, h, score))
    return dedup_boxes(boxes, iou_threshold)


def poll_once(
    pool: SharedFramePool,
    backend,
    results: queue.Queue,
    last_sequence: int,
) -> FrameMetadata | None:
    """Process at most one frame newer than last_sequence; return its metadata."""
    metadata = pool.latest_after(last_sequence)
    if metadata is None:
        return None
    payload = pool.copy(metadata)
    if payload is None:
        return None
    frame_rgb = yuyv_to_rgb(payload, pool.width, pool.height)
    for x, y, w, h, _score in detect_frame(backend, frame_rgb):
        while True:
            try:
                results.put_nowait((metadata.realtime_ns, x, y, w, h))
                break
            except queue.Full:
                try:
                    results.get_nowait()
                except queue.Empty:
                    pass
    return metadata


def run_mine_detection(
    results: queue.Queue,
    backend,
    *,
    stop: threading.Event | None = None,
    pool_name: str = "cm2",
    period_s: float = DEFAULT_PERIOD_S,
    log=lambda text: print(text, file=sys.stderr, flush=True),
) -> None:
    """Feed ``results`` for the mission's lifetime; returns when stop is set.

    Attaches to the sensing frame pool and reattaches if sensing restarts.
    Never raises out of the loop: pool loss and backend faults are logged and
    retried so the mission thread that runs this cannot die mid-flight.
    """
    stop = stop or threading.Event()
    pool = None
    last_sequence = 0
    last_progress = time.monotonic()
    while not stop.is_set():
        began = time.monotonic()
        wait_s = period_s
        try:
            if pool is None:
                pool = SharedFramePool.attach(pool_name)
                last_sequence = 0
                last_progress = time.monotonic()
                log(f"mine_detector: attached to pool '{pool_name}'")
            metadata = poll_once(pool, backend, results, last_sequence)
            if metadata is not None:
                last_sequence = metadata.sequence
                last_progress = time.monotonic()
            else:
                # No new frame yet; at period 0 this must not busy-poll.
                wait_s = max(period_s, 0.02)
                if time.monotonic() - last_progress > STALE_POOL_S:
                    # Sensing restarted or died: our mapping points at
                    # unlinked files and will never advance.  Reattach.
                    log("mine_detector: pool stale, reattaching")
                    pool.close()
                    pool = None
        except (RuntimeError, OSError) as exc:
            log(f"mine_detector: waiting for sensing ({exc})")
            wait_s = max(period_s, 1.0)
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    pass
                pool = None
        stop.wait(max(0.0, wait_s - (time.monotonic() - began)))
    if pool is not None:
        pool.close()


def get_hailo_bounding_boxes(results: queue.Queue, **kwargs) -> None:
    """Mission-runner entry point; blocks, so run it in a dedicated thread."""
    run_mine_detection(results, HailoYoloBackend(), **kwargs)
