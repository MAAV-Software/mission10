"""Shadow forward-depth monitor over the OV9281 shared frame pool.

Runs Depth Anything V2 on the forward camera and logs what an obstacle gate
WOULD have done.  It publishes nothing to flight control and holds no camera:
the flight recorder owns the OV9281 and mirrors captured frames into the
``ov9281`` shared pool; this consumer attaches read-only, exactly like the
mine detector does with the ``cm2`` pool.

Depth Anything output is relative inverse depth with an arbitrary per-frame
scale, so no fixed "meters" threshold exists.  Each sample therefore logs a
scale-free looming ratio: the 95th-percentile nearness inside the flight
corridor divided by the whole-frame median.  A tall ratio means something in
the corridor is much closer than the general scene.  The trigger threshold is
chosen later from recorded shadow flights, not guessed here.

Each processed frame appends one JSON line to the log file:

    {"t_ns": ..., "sequence": ..., "corridor_p95": ..., "frame_median": ...,
     "loom_ratio": ..., "latency_ms": ...}
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Callable

import numpy as np

from .shared_frame_pool import SharedFramePool

POOL_NAME = "ov9281"
STALE_POOL_S = 5.0
CORRIDOR_X = (1 / 3, 2 / 3)
CORRIDOR_Y = (1 / 4, 3 / 4)


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = (np.arange(height) * image.shape[0] // height).clip(0, image.shape[0] - 1)
    cols = (np.arange(width) * image.shape[1] // width).clip(0, image.shape[1] - 1)
    return image[rows][:, cols]


def loom_sample(nearness: np.ndarray) -> dict:
    """Scale-free corridor statistics from one relative inverse-depth map."""
    height, width = nearness.shape
    corridor = nearness[
        int(height * CORRIDOR_Y[0]) : int(height * CORRIDOR_Y[1]),
        int(width * CORRIDOR_X[0]) : int(width * CORRIDOR_X[1]),
    ]
    corridor_p95 = float(np.percentile(corridor, 95))
    frame_median = float(np.median(nearness))
    scale = max(abs(frame_median), 1e-6)
    return {
        "corridor_p95": corridor_p95,
        "frame_median": frame_median,
        "loom_ratio": corridor_p95 / scale,
    }


class DepthAnythingBackend:
    """Depth Anything V2 HEF on the shared Hailo VDevice."""

    def __init__(self, hef_path) -> None:
        from .hailo_device import shared_vdevice

        self.vdevice = shared_vdevice()
        self.infer_model = self.vdevice.create_infer_model(str(hef_path))
        self.infer_model.set_batch_size(1)
        height, width, channels = self.infer_model.input().shape
        if channels != 3:
            raise RuntimeError(f"expected an RGB depth HEF, got shape "
                               f"{(height, width, channels)}")
        self.input_size = (height, width)
        self.output_buffer = np.empty(
            tuple(self.infer_model.output().shape), dtype=np.float32
        )
        self.configured = self.infer_model.configure()
        self.configured.__enter__()

    def infer(self, gray: np.ndarray) -> np.ndarray:
        """Mono frame in, relative inverse depth (model resolution) out."""
        small = resize_nearest(gray, *self.input_size)
        rgb = np.ascontiguousarray(np.stack([small] * 3, axis=-1))
        bindings = self.configured.create_bindings()
        bindings.input().set_buffer(rgb)
        bindings.output().set_buffer(self.output_buffer)
        self.configured.run([bindings], timeout=2000)
        return self.output_buffer.reshape(self.input_size)

    def close(self) -> None:
        self.configured.__exit__(None, None, None)


def run_depth_monitor(
    backend,
    log_path,
    *,
    stop: threading.Event | None = None,
    pool_name: str = POOL_NAME,
    period_s: float = 0.0,
    on_sample: Callable[[dict], None] | None = None,
    log=lambda text: print(text, file=sys.stderr, flush=True),
) -> None:
    """Blocking shadow loop; run in a dedicated thread, stops via ``stop``."""
    stop = stop or threading.Event()
    pool = None
    last_sequence = 0
    last_progress = time.monotonic()
    with open(log_path, "a", buffering=1) as sink:
        while not stop.is_set():
            began = time.monotonic()
            wait_s = period_s
            try:
                if pool is None:
                    pool = SharedFramePool.attach(pool_name)
                    last_sequence = 0
                    last_progress = time.monotonic()
                    log(f"depth_monitor: attached to pool '{pool_name}'")
                metadata = pool.latest_after(last_sequence)
                payload = pool.copy(metadata) if metadata is not None else None
                if payload is not None:
                    gray = np.frombuffer(payload, dtype=np.uint8).reshape(
                        pool.height, pool.width
                    )
                    t0 = time.perf_counter()
                    nearness = backend.infer(gray)
                    sample = loom_sample(nearness)
                    sample["t_ns"] = metadata.realtime_ns
                    sample["sequence"] = metadata.sequence
                    sample["latency_ms"] = (time.perf_counter() - t0) * 1e3
                    sink.write(json.dumps(sample, separators=(",", ":")) + "\n")
                    if on_sample is not None:
                        on_sample(sample)
                    last_sequence = metadata.sequence
                    last_progress = time.monotonic()
                else:
                    wait_s = max(period_s, 0.02)
                    if time.monotonic() - last_progress > STALE_POOL_S:
                        log("depth_monitor: pool stale, reattaching")
                        pool.close()
                        pool = None
            except (RuntimeError, OSError) as exc:
                log(f"depth_monitor: waiting for OV9281 pool ({exc})")
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
