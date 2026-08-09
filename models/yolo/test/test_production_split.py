import json
import unittest
from pathlib import Path


class TestProductionSplit(unittest.TestCase):
    def test_is_complete_disjoint_and_pilot_safe(self):
        path = Path(__file__).parents[1] / "train" / "production300-split.json"
        data = json.loads(path.read_text())
        self.assertEqual(data["schema"], "mission10-yolo-split/1")
        self.assertEqual(data["seed"], "m10")
        scenes = data["scenes"]
        self.assertEqual(
            {name: len(values) for name, values in scenes.items()},
            {"train": 240, "val": 30, "test": 30},
        )
        for values in scenes.values():
            self.assertEqual(values, sorted(set(values)))
        owners = {
            scene: split
            for split, values in scenes.items()
            for scene in values
        }
        self.assertEqual(set(owners), set(range(300)))
        self.assertTrue(set(range(40)).issubset(scenes["train"]))
        self.assertFalse(set(range(40)).intersection(scenes["val"]))
        self.assertFalse(set(range(40)).intersection(scenes["test"]))


if __name__ == "__main__":
    unittest.main()
