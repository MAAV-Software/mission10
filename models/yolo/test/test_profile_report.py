import json
import tempfile
import unittest
from pathlib import Path

from export.profile_report import compare, summarize


def _report(path: Path, fps: float, classes: int) -> Path:
    stats = {
        "profiler_parameters": {
            "compiler_version": "3.34.0",
            "profiling_mode": "Compiled",
            "is_quantized": True,
        },
        "model_details": {
            "model_name": "yolov11m",
            "hw_arch": "hailo8",
            "weights": 20_000_000,
            "total_ops_per_frame": 68_000_000_000,
            "input_shapes": ["640x640x3"],
            "output_shapes": [f"{classes}x5x100"],
            "post_processing": ["yolov8 NMS"],
        },
        "performance_details": [
            {
                "batch_size": 1,
                "fps": fps,
                "latency": 1000 / fps,
                "number_of_contexts": 4,
                "input_bw": 10,
                "output_bw": 20,
                "ops_per_second": 30,
                "measured_mac_util": 40,
            }
        ],
    }
    path.write_text(
        "<html><script>const report = "
        + json.dumps({"csv_data": "", "stats": stats})
        + ";</script></html>"
    )
    return path


class TestProfilerReport(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_extracts_embedded_summary(self):
        summary = summarize(_report(self.root / "report.html", 50.0, 1))
        self.assertEqual(summary["compiler_version"], "3.34.0")
        self.assertEqual(summary["output_shapes"], ["1x5x100"])
        self.assertEqual(summary["batch1"]["fps"], 50.0)

    def test_compares_numeric_metrics(self):
        baseline = summarize(_report(self.root / "base.html", 50.0, 80))
        candidate = summarize(_report(self.root / "candidate.html", 60.0, 1))
        result = compare(baseline, candidate)
        self.assertEqual(result["batch1"]["fps"]["absolute"], 10.0)
        self.assertEqual(result["batch1"]["fps"]["percent"], 20.0)
        self.assertEqual(
            result["model"]["output_shapes"],
            {"baseline": ["80x5x100"], "candidate": ["1x5x100"]},
        )


if __name__ == "__main__":
    unittest.main()
