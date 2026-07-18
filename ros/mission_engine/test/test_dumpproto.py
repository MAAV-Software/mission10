import unittest

from mission_engine.core.dumpproto import build_payload, decode_frame, encode_frame
from mission_engine.core.minelog import DetectionObs, MineLog


class TestDumpProto(unittest.TestCase):
    def _log(self):
        log = MineLog(gate_m=1.0, confirm_obs=2, confirm_passes=1)
        for i in range(3):
            log.ingest(
                DetectionObs(
                    t=0.1 * i,
                    ground_local=(2.0, 3.0),
                    conf=0.9,
                    class_id="mine",
                    ll=(42.29, -83.71),
                )
            )
        return log

    def test_payload_schema(self):
        p = build_payload(self._log(), "drone2", "m1", 0.0, 100.0)
        self.assertEqual(p["schema"], "minefield-dump/1")
        self.assertEqual(len(p["mines"]), 1)
        m = p["mines"][0]
        self.assertEqual(m["id"], "drone2/0")
        self.assertEqual(m["status"], "confirmed")
        self.assertEqual(m["n_obs"], 3)
        self.assertEqual(m["ll"], [42.29, -83.71])
        self.assertEqual(m["ll_per_pass"], [[42.29, -83.71]])
        self.assertIsNone(m["tag_id"])

    def test_frame_roundtrip(self):
        p = build_payload(self._log(), "drone2", "m1", 0.0, 100.0)
        data = encode_frame(p) + b"trailing"
        out, rest = decode_frame(data)
        self.assertEqual(out, p)
        self.assertEqual(rest, b"trailing")

    def test_rejects_unknown_schema(self):
        bad = encode_frame({"schema": "minefield-dump/2", "mines": []})
        with self.assertRaises(ValueError):
            decode_frame(bad)

    def test_rejects_short_frame(self):
        p = build_payload(self._log(), "drone2", "m1", 0.0, 100.0)
        with self.assertRaises(ValueError):
            decode_frame(encode_frame(p)[:-5])


if __name__ == "__main__":
    unittest.main()
