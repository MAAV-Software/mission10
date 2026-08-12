"""Unit tests for the mine detection pool consumer.

Backend inference is faked; these tests cover tiling, seam dedup, coordinate
remapping, and the caller-owned result queue contract.  HailoRT parsing and
the real HEF are validated on an aircraft, not here.
"""
from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from sensing.mine_detector import (
    HailoYoloBackend,
    TILE,
    dedup_boxes,
    detect_frame,
    poll_once,
    tile_origins,
)
from sensing.shared_frame_pool import SharedFramePool, segment_paths


WIDTH = 1640
HEIGHT = 1232


class TileOriginsTest(unittest.TestCase):
    def test_cm2_frame_yields_six_covering_tiles(self):
        origins = tile_origins(WIDTH, HEIGHT)
        self.assertEqual(len(origins), 6)
        for x0, y0 in origins:
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x0 + TILE, WIDTH)
            self.assertLessEqual(y0 + TILE, HEIGHT)
        self.assertIn((0, 0), origins)
        self.assertIn((WIDTH - TILE, HEIGHT - TILE), origins)
        columns = sorted({x0 for x0, _ in origins})
        for left, right in zip(columns, columns[1:]):
            self.assertLess(right - left, TILE, "columns must overlap")

    def test_small_frame_is_one_tile(self):
        self.assertEqual(tile_origins(320, 240), [(0, 0)])


class DedupTest(unittest.TestCase):
    def test_seam_duplicates_collapse_to_strongest(self):
        duplicates = [
            (100.0, 100.0, 40.0, 40.0, 0.9),
            (102.0, 101.0, 40.0, 40.0, 0.6),
        ]
        distinct = (600.0, 600.0, 40.0, 40.0, 0.8)
        kept = dedup_boxes(duplicates + [distinct])
        self.assertEqual(len(kept), 2)
        self.assertIn(duplicates[0], kept)
        self.assertIn(distinct, kept)


class RecordingBackend:
    """Reports one fixed box in every tile whose view is nonzero at (10, 10)."""

    def __init__(self):
        self.tiles_seen = 0

    def infer(self, tile_rgb):
        self.tiles_seen += 1
        if tile_rgb[10, 10].any():
            return [(10.0, 10.0, 20.0, 20.0, 0.9)]
        return []


class DetectFrameTest(unittest.TestCase):
    def test_remaps_tile_boxes_into_frame_pixels(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :] = 255
        backend = RecordingBackend()
        boxes = detect_frame(backend, frame)
        self.assertEqual(backend.tiles_seen, 6)
        expected = {(x0 + 10.0, y0 + 10.0) for x0, y0 in tile_origins(WIDTH, HEIGHT)}
        self.assertEqual({(x, y) for x, y, _w, _h, _s in boxes}, expected)


class HailoParseTest(unittest.TestCase):
    def test_nms_by_class_buffer_decodes_to_tile_pixels(self):
        parser = HailoYoloBackend.__new__(HailoYoloBackend)
        parser.confidence = 0.3
        parser.tile = TILE
        buffer = np.array(
            [2.0,
             0.10, 0.20, 0.30, 0.40, 0.90,
             0.50, 0.50, 0.60, 0.60, 0.10],
            dtype=np.float32,
        )
        boxes = parser._parse_nms(buffer)
        self.assertEqual(len(boxes), 1)
        x, y, w, h, score = boxes[0]
        self.assertAlmostEqual(x, 0.20 * TILE, places=3)
        self.assertAlmostEqual(y, 0.10 * TILE, places=3)
        self.assertAlmostEqual(w, 0.20 * TILE, places=3)
        self.assertAlmostEqual(h, 0.20 * TILE, places=3)
        self.assertAlmostEqual(score, 0.90, places=5)


class PollOnceTest(unittest.TestCase):
    """The result queue contract against a real shared pool, in one process."""

    POOL = "cm2_mine_test"
    STAMP_NS = 1_784_364_300_500_000_000

    def setUp(self):
        for path in segment_paths(self.POOL):
            path.unlink(missing_ok=True)
        self.owner = SharedFramePool.create(self.POOL, WIDTH, HEIGHT, slots=4)
        self.timestamp_lock = threading.Lock()
        self.mission_node = SimpleNamespace(latest_timestamp=self.STAMP_NS)

    def tearDown(self):
        self.reader.close()
        self.owner.close()

    def publish_gray(self, level):
        slot = self.owner.begin_write()
        buffer = np.ndarray((HEIGHT, WIDTH, 2), dtype=np.uint8, buffer=slot.buffer)
        buffer[:, :, 0] = level  # luma
        buffer[:, :, 1] = 128  # unsigned chroma zero
        return slot.commit(
            sensor_boottime_ns=1,
            realtime_ns=self.STAMP_NS,
            exposure_us=100,
            analogue_gain=1.0,
        )

    def poll(self, backend, results, last_sequence=0):
        return poll_once(
            self.reader,
            self.timestamp_lock,
            backend,
            self.mission_node,
            results,
            last_sequence,
        )

    def test_one_queue_item_per_detected_box(self):
        published = self.publish_gray(255)
        self.reader = SharedFramePool.attach(self.POOL)
        results = queue.Queue()
        backend = RecordingBackend()
        metadata = self.poll(backend, results)
        self.assertEqual(metadata.sequence, published.sequence)
        self.assertEqual(results.qsize(), 6)
        while not results.empty():
            stamp_ns, x, y, w, h = results.get_nowait()
            self.assertEqual(stamp_ns, self.STAMP_NS)
            self.assertEqual((w, h), (20.0, 20.0))
        # Nothing newer: a second poll must not queue anything.
        self.assertIsNone(self.poll(backend, results, metadata.sequence))
        self.assertTrue(results.empty())

    def test_dark_frame_queues_nothing(self):
        self.publish_gray(0)
        self.reader = SharedFramePool.attach(self.POOL)
        results = queue.Queue()
        metadata = self.poll(RecordingBackend(), results)
        self.assertIsNotNone(metadata)
        self.assertTrue(results.empty())


if __name__ == "__main__":
    unittest.main()
