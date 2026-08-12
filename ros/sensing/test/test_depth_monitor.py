"""Unit tests for the shadow depth monitor (fake inference backend)."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from sensing.depth_monitor import (
    loom_sample,
    resize_nearest,
    run_depth_monitor,
)
from sensing.shared_frame_pool import SharedFramePool, segment_paths

WIDTH = 1280
HEIGHT = 800


class ResizeTest(unittest.TestCase):
    def test_downscale_keeps_quadrant_structure(self):
        image = np.zeros((800, 1280), dtype=np.uint8)
        image[:400, :640] = 200
        small = resize_nearest(image, 100, 160)
        self.assertEqual(small.shape, (100, 160))
        self.assertEqual(small[10, 10], 200)
        self.assertEqual(small[90, 150], 0)


class LoomSampleTest(unittest.TestCase):
    def test_near_corridor_blob_raises_ratio(self):
        flat = np.full((100, 160), 10.0, dtype=np.float32)
        quiet = loom_sample(flat)
        self.assertAlmostEqual(quiet["loom_ratio"], 1.0, places=3)

        blob = flat.copy()
        blob[40:60, 70:90] = 80.0  # near object in the corridor
        loud = loom_sample(blob)
        self.assertGreater(loud["loom_ratio"], 5.0)
        self.assertAlmostEqual(loud["frame_median"], 10.0, places=3)

    def test_near_object_outside_corridor_stays_quiet(self):
        edge = np.full((100, 160), 10.0, dtype=np.float32)
        edge[:, :20] = 80.0  # near object far left, out of the corridor
        self.assertAlmostEqual(loom_sample(edge)["loom_ratio"], 1.0, places=3)


class FakeDepthBackend:
    """Nearness proportional to pixel brightness."""

    def infer(self, gray):
        return gray.astype(np.float32)


class MonitorLoopTest(unittest.TestCase):
    POOL = "ov9281_depth_test"

    def setUp(self):
        for path in segment_paths(self.POOL):
            path.unlink(missing_ok=True)
        self.owner = SharedFramePool.create(
            self.POOL, WIDTH, HEIGHT, bytes_per_pixel=1
        )

    def tearDown(self):
        self.owner.close()

    def publish(self, frame):
        slot = self.owner.begin_write()
        destination = np.ndarray((HEIGHT, WIDTH), dtype=np.uint8, buffer=slot.buffer)
        np.copyto(destination, frame)
        return slot.commit(
            sensor_boottime_ns=1,
            realtime_ns=1_784_364_300_500_000_000,
            exposure_us=100,
            analogue_gain=1.0,
        )

    def test_one_frame_produces_one_log_line(self):
        frame = np.full((HEIGHT, WIDTH), 10, dtype=np.uint8)
        frame[300:500, 500:700] = 250  # bright = near, inside the corridor
        self.publish(frame)

        stop = threading.Event()
        samples = []

        def on_sample(sample):
            samples.append(sample)
            stop.set()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "depth_shadow.jsonl"
            worker = threading.Thread(
                target=run_depth_monitor,
                args=(FakeDepthBackend(), log_path),
                kwargs={
                    "stop": stop,
                    "pool_name": self.POOL,
                    "on_sample": on_sample,
                    "log": lambda text: None,
                },
            )
            worker.start()
            worker.join(timeout=10.0)
            self.assertFalse(worker.is_alive())
            lines = log_path.read_text().splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(len(samples), 1)
        logged = json.loads(lines[0])
        self.assertEqual(logged["t_ns"], 1_784_364_300_500_000_000)
        self.assertEqual(logged["sequence"], 1)
        self.assertGreater(logged["loom_ratio"], 5.0)
        self.assertIn("latency_ms", logged)


if __name__ == "__main__":
    unittest.main()
