import unittest

from mission_engine.core.minelog import (
    CANDIDATE,
    CONFIRMED,
    VERIFIED,
    DetectionObs,
    MineLog,
)


def obs(t, n, e, conf=0.9, ll=None, tag_id=None):
    return DetectionObs(
        t=t, ground_local=(n, e), conf=conf, class_id="mine", ll=ll, tag_id=tag_id
    )


class TestMineLog(unittest.TestCase):
    def test_gate_splits_and_joins(self):
        log = MineLog(gate_m=1.0)
        log.ingest(obs(0.0, 0.0, 0.0))
        log.ingest(obs(0.1, 0.3, 0.0))  # inside gate -> same cluster
        log.ingest(obs(0.2, 5.0, 0.0))  # far -> new cluster
        self.assertEqual(len(log.clusters), 2)
        self.assertEqual(log.clusters[0].n_obs, 2)

    def test_weighted_centroid_and_spread(self):
        log = MineLog(gate_m=2.0)
        log.ingest(obs(0.0, 0.0, 0.0, conf=1.0))
        c = log.ingest(obs(0.1, 1.0, 0.0, conf=1.0))
        self.assertAlmostEqual(c.centroid[0], 0.5, places=9)
        self.assertGreater(c.spread_m, 0.0)

    def test_pass_counting_and_confirmation(self):
        log = MineLog(gate_m=1.0, pass_gap_s=5.0, confirm_obs=4, confirm_passes=2)
        for i in range(3):
            c = log.ingest(obs(0.1 * i, 0.0, 0.0, ll=(42.0, -83.0)))
        self.assertEqual(c.status, CANDIDATE)
        self.assertEqual(c.n_passes, 1)
        c = log.ingest(obs(20.0, 0.1, 0.0, ll=(42.000001, -83.0)))  # new pass
        self.assertEqual(c.n_passes, 2)
        self.assertEqual(c.status, CONFIRMED)
        # first pass's ll mean was banked when the second pass opened
        self.assertEqual(len(c.ll_per_pass), 1)
        log.finalize()
        self.assertEqual(len(c.ll_per_pass), 2)
        self.assertAlmostEqual(c.ll[0], 42.0000005, places=9)

    def test_tag_verifies(self):
        log = MineLog()
        c = log.ingest(obs(0.0, 0.0, 0.0, tag_id="tag36h11:12"))
        self.assertEqual(c.status, VERIFIED)
        self.assertEqual(c.tag_ids, ["tag36h11:12"])
        self.assertIsNone(log.next_dip_target())  # verified needs no dip


if __name__ == "__main__":
    unittest.main()
