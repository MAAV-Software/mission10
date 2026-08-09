"""Tests for the Mission 10 minefield pathfinder."""

import unittest

import iarc_pathfinder as pathfinder


REFERENCE_MINES = {
    (1, 63), (2, 40), (4, 55), (5, 78), (7, 11),
    (10, 115), (13, 132), (16, 32), (17, 88), (20, 11),
    (21, 106), (23, 121), (26, 75), (27, 43), (29, 27),
    (30, 112), (31, 131), (32, 101), (34, 55), (36, 84),
}


class PathAssertions:
    def assert_valid_path(self, path, mines, G):
        self.assertIsNotNone(path)
        self.assertEqual(path[0][1], 0)
        self.assertEqual(path[-1][1], pathfinder.ROWS - 1)

        for x, _ in path:
            self.assertGreaterEqual(x, G)
            self.assertLess(x, pathfinder.COLS - G)

        for first, second in zip(path, path[1:]):
            distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
            self.assertEqual(distance, 1)

        self.assertFalse(set(path) & mines)
        self.assertFalse(pathfinder.compute_green_zone(path, G) & mines)


class ClearanceMapTests(unittest.TestCase):
    def test_single_mine_uses_chebyshev_distance(self):
        mine = (11, 47)

        clearance = pathfinder.compute_clearance_map({mine})

        for cell in ((11, 47), (12, 47), (11, 52), (4, 50), (39, 149)):
            expected = max(abs(cell[0] - mine[0]), abs(cell[1] - mine[1]))
            self.assertEqual(clearance[cell], expected)

    def test_multiple_mines_match_direct_distance_calculation(self):
        mines = {(3, 4), (19, 72), (35, 140)}

        clearance = pathfinder.compute_clearance_map(mines)

        for x in range(pathfinder.COLS):
            for y in range(pathfinder.ROWS):
                expected = min(
                    max(abs(x - mine_x), abs(y - mine_y))
                    for mine_x, mine_y in mines
                )
                self.assertEqual(clearance[(x, y)], expected)

    def test_empty_field_has_effectively_infinite_clearance(self):
        clearance = pathfinder.compute_clearance_map(set())

        self.assertEqual(len(clearance), pathfinder.COLS * pathfinder.ROWS)
        self.assertGreater(
            min(clearance.values()),
            max(pathfinder.COLS, pathfinder.ROWS),
        )

    def test_out_of_bounds_mine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the grid"):
            pathfinder.compute_clearance_map({(pathfinder.COLS, 10)})


class GridIndexTests(unittest.TestCase):
    def test_clamps_coordinates_to_grid(self):
        self.assertEqual(pathfinder.grid_index(-0.1, 0.0, 1.0, 40), 0)
        self.assertEqual(pathfinder.grid_index(0.0, 0.0, 1.0, 40), 0)
        self.assertEqual(pathfinder.grid_index(17.5, 0.0, 1.0, 40), 17)
        self.assertEqual(pathfinder.grid_index(40.0, 0.0, 1.0, 40), 39)
        self.assertEqual(pathfinder.grid_index(50.0, 0.0, 1.0, 40), 39)

    def test_generate_minefield_maps_arena_corners(self):
        with open(pathfinder.constants_path, "r") as bounds_file:
            bounds = [tuple(map(float, line.split())) for line in bounds_file]
        lats = [lat for lat, _ in bounds]
        lons = [lon for _, lon in bounds]

        mines = pathfinder.generate_minefield(
            [(min(lats), min(lons)), (max(lats), max(lons))]
        )

        self.assertEqual(mines, {(0, 0), (pathfinder.COLS - 1, pathfinder.ROWS - 1)})


class ShortestSafePathTests(PathAssertions, unittest.TestCase):
    def test_empty_field_is_straight_for_every_width(self):
        clearance = pathfinder.compute_clearance_map(set())

        for G in range((pathfinder.COLS - 1) // 2 + 1):
            path = pathfinder.shortest_safe_path(clearance, G)
            self.assertEqual(len(path), pathfinder.ROWS)
            self.assertEqual(len({x for x, _ in path}), 1)

    def test_mine_wall_has_no_path(self):
        mines = {(x, 70) for x in range(pathfinder.COLS)}
        clearance = pathfinder.compute_clearance_map(mines)

        self.assertIsNone(pathfinder.shortest_safe_path(clearance, 0))

    def test_path_uses_gap_and_keeps_requested_margin(self):
        mines = {(x, 70) for x in range(pathfinder.COLS) if x != 20}
        clearance = pathfinder.compute_clearance_map(mines)

        path = pathfinder.shortest_safe_path(clearance, 0)

        self.assert_valid_path(path, mines, 0)
        self.assertIn((20, 70), path)


class BestPathTests(PathAssertions, unittest.TestCase):
    def test_empty_field_returns_widest_straight_corridor(self):
        path, G, clearance = pathfinder.find_best_path(set())

        self.assertEqual(G, (pathfinder.COLS - 1) // 2)
        self.assertEqual(len(path), pathfinder.ROWS)
        self.assertGreater(clearance, pathfinder.ROWS)

    def test_result_is_safe_and_has_best_width_length_score(self):
        mines = {
            (5, 20),
            (33, 35),
            (10, 60),
            (28, 80),
            (17, 105),
            (35, 130),
        }
        path, best_G, _ = pathfinder.find_best_path(mines)
        clearance = pathfinder.compute_clearance_map(mines)

        self.assert_valid_path(path, mines, best_G)
        chosen = pathfinder.score_path(path, best_G, mines)["score"]

        candidate_scores = []
        for G in range((pathfinder.COLS - 1) // 2 + 1):
            candidate = pathfinder.shortest_safe_path(clearance, G)
            if candidate is not None:
                candidate_scores.append(
                    pathfinder.score_path(candidate, G, mines)["score"]
                )

        self.assertAlmostEqual(chosen, max(candidate_scores))

    def test_reference_field_regression(self):
        path, G, _ = pathfinder.find_best_path(REFERENCE_MINES)
        result = pathfinder.score_path(path, G, REFERENCE_MINES)

        self.assertEqual(G, 4)
        self.assertEqual(len(path), 157)
        self.assertAlmostEqual(result["score"], 171.97452229299364)
        self.assertEqual(result["mines_on_path"], 0)
        self.assertEqual(result["mines_in_green"], 0)

    def test_reference_path_commands_round_trip(self):
        path, G, _ = pathfinder.find_best_path(REFERENCE_MINES)

        command_lines = pathfinder.grid_path_to_commands(path, G).splitlines()

        start, *move_commands = command_lines
        _, start_x, encoded_G = start.split(",")
        self.assertEqual(int(start_x), path[0][0])
        self.assertEqual(int(encoded_G), G)
        self.assertTrue(move_commands[0].startswith("U,"))
        self.assertTrue(move_commands[-1].startswith("U,"))

        delta_for_direction = {
            "U": (0, 1),
            "D": (0, -1),
            "R": (1, 0),
            "L": (-1, 0),
        }
        decoded_path = [path[0]]
        for command in move_commands:
            direction, raw_count = command.split(",")
            dx, dy = delta_for_direction[direction]
            for _ in range(int(raw_count)):
                x, y = decoded_path[-1]
                decoded_path.append((x + dx, y + dy))

        self.assertEqual(decoded_path, path)


class CommandEncodingTests(unittest.TestCase):
    def test_rejects_path_without_a_move(self):
        with self.assertRaisesRegex(ValueError, "at least one move"):
            pathfinder.grid_path_to_commands([(2, 0)])

    def test_rejects_lateral_first_move(self):
        with self.assertRaisesRegex(ValueError, "begin and end"):
            pathfinder.grid_path_to_commands([(2, 0), (3, 0), (3, 1)])

    def test_rejects_non_adjacent_move(self):
        with self.assertRaisesRegex(ValueError, "non-adjacent"):
            pathfinder.grid_path_to_commands([(2, 0), (2, 2)])


if __name__ == "__main__":
    unittest.main()
