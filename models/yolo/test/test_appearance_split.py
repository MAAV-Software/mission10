import json
import unittest
from pathlib import Path


class TestAppearanceSplit(unittest.TestCase):
    def test_is_complete_disjoint_and_has_the_approved_sizes(self):
        path = Path(__file__).parents[1] / "train" / "appearance60-split.json"
        data = json.loads(path.read_text())
        self.assertEqual(data["schema"], "mission10-yolo-split/1")
        self.assertEqual(data["seed"], "m10-appearance-v1")
        scenes = data["scenes"]
        self.assertEqual(
            {name: len(values) for name, values in scenes.items()},
            {"train": 48, "val": 6, "test": 6},
        )
        for values in scenes.values():
            self.assertEqual(values, sorted(set(values)))
        owners = {
            scene: split
            for split, values in scenes.items()
            for scene in values
        }
        self.assertEqual(set(owners), set(range(60)))


if __name__ == "__main__":
    unittest.main()
