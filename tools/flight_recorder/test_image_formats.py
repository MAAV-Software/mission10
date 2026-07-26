#!/usr/bin/env python3

import unittest

import numpy as np

from image_formats import YUYV_BYTES_PER_PIXEL, pack_yuyv_frame


class PackYuyvFrameTest(unittest.TestCase):
    def test_removes_row_padding_without_reordering_pixels(self):
        width, height, padded_width = 4, 2, 6
        frame = np.arange(
            height * padded_width * YUYV_BYTES_PER_PIXEL, dtype=np.uint8
        ).reshape(height, padded_width, YUYV_BYTES_PER_PIXEL)

        packed = pack_yuyv_frame(frame, width, height)

        self.assertEqual(len(packed), width * height * YUYV_BYTES_PER_PIXEL)
        np.testing.assert_array_equal(
            np.frombuffer(packed, dtype=np.uint8).reshape(height, width, 2),
            frame[:, :width, :],
        )

    def test_accepts_tight_frame(self):
        frame = np.zeros((2, 4, 2), dtype=np.uint8)
        self.assertEqual(len(pack_yuyv_frame(frame, 4, 2)), 16)

    def test_rejects_short_or_malformed_frame(self):
        with self.assertRaises(ValueError):
            pack_yuyv_frame(np.zeros((2, 3, 2), dtype=np.uint8), 4, 2)
        with self.assertRaises(ValueError):
            pack_yuyv_frame(np.zeros((2, 4), dtype=np.uint8), 4, 2)
        with self.assertRaises(ValueError):
            pack_yuyv_frame(np.zeros((2, 4, 2), dtype=np.uint16), 4, 2)


if __name__ == "__main__":
    unittest.main()
