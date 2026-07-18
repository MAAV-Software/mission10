import unittest

from mission_engine.core.tiles import tile_grid

W, H, TILE, OV = 1640, 1232, 640, 192


class TestTileGrid(unittest.TestCase):
    def test_cm2_frame_shape(self):
        grid = tile_grid(W, H, TILE, OV)
        self.assertEqual(len(grid), 12)  # 4 columns x 3 rows
        xs = sorted({x for x, _ in grid})
        ys = sorted({y for _, y in grid})
        self.assertEqual((xs[0], xs[-1]), (0, W - TILE))
        self.assertEqual((ys[0], ys[-1]), (0, H - TILE))

    def test_in_bounds_and_covering(self):
        grid = tile_grid(W, H, TILE, OV)
        covered_x = [False] * W
        covered_y = [False] * H
        for x, y in grid:
            self.assertTrue(0 <= x <= W - TILE and 0 <= y <= H - TILE)
            for u in range(x, x + TILE):
                covered_x[u] = True
            for v in range(y, y + TILE):
                covered_y[v] = True
        self.assertTrue(all(covered_x) and all(covered_y))

    def test_overlap_at_least_requested(self):
        for x0, x1 in zip(xs := sorted({x for x, _ in tile_grid(W, H, TILE, OV)}), xs[1:]):
            self.assertLessEqual(x1 - x0, TILE - OV)
        for y0, y1 in zip(ys := sorted({y for _, y in tile_grid(W, H, TILE, OV)}), ys[1:]):
            self.assertLessEqual(y1 - y0, TILE - OV)

    def test_whole_object_guarantee(self):
        # any object with size <= overlap fits whole in some tile, wherever
        # it sits; scan every position of a worst-case object
        xs = sorted({x for x, _ in tile_grid(W, H, TILE, OV)})
        for a in range(0, W - OV + 1):
            self.assertTrue(
                any(x <= a and a + OV <= x + TILE for x in xs),
                f"object at x={a} spans no tile whole",
            )

    def test_small_frame_single_tile(self):
        self.assertEqual(tile_grid(640, 480, TILE, OV), [(0, 0)])

    def test_deterministic(self):
        self.assertEqual(tile_grid(W, H, TILE, OV), tile_grid(W, H, TILE, OV))

    def test_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            tile_grid(W, H, tile=640, overlap=640)
        with self.assertRaises(ValueError):
            tile_grid(0, H)


if __name__ == "__main__":
    unittest.main()
