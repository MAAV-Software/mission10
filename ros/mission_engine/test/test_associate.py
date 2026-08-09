import itertools
import math
import random
import unittest

from mission_engine.core.associate import (
    Associator,
    Tracklet,
    apply_se2,
    blind_align,
    form_tracklets,
    hungarian,
    rigid_fit,
)
from mission_engine.core.minelog import DetectionObs


def obs(t, n, e, conf=1.0, class_id="mine", tag_id=None):
    return DetectionObs(t=t, ground_local=(n, e), conf=conf, class_id=class_id, tag_id=tag_id)


def tracklet(n, e, class_id="mine", t0=0.0, t1=1.0, weight=1.0, tags=()):
    return Tracklet(
        center=(n, e), t0=t0, t1=t1, n_obs=5, weight=weight,
        spread_m=0.05, class_id=class_id, tag_ids=list(tags),
    )


class TestHungarian(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(7)
        for _ in range(20):
            n = rng.randint(2, 5)
            cost = [[rng.uniform(0, 10) for _ in range(n)] for _ in range(n)]
            assign = hungarian(cost)
            got = sum(cost[i][assign[i]] for i in range(n))
            best = min(
                sum(cost[i][p[i]] for i in range(n))
                for p in itertools.permutations(range(n))
            )
            self.assertAlmostEqual(got, best, places=9)
            self.assertEqual(sorted(assign), list(range(n)))


class TestGeometry(unittest.TestCase):
    def test_rigid_fit_recovers_se2(self):
        src = [(0.0, 0.0), (4.0, 1.0), (2.0, 5.0), (7.0, 3.0)]
        theta, trans = 0.3, (1.5, -2.0)
        dst = [apply_se2(theta, trans, p) for p in src]
        th, tr = rigid_fit(src, dst)
        self.assertAlmostEqual(th, theta, places=9)
        self.assertAlmostEqual(tr[0], trans[0], places=9)
        self.assertAlmostEqual(tr[1], trans[1], places=9)

    def test_blind_align_recovers_common_mode(self):
        """Field + offset/yaw within the coarse gate, no correspondences
        given: the alignment loop must recover the transform."""
        rng = random.Random(1)
        field = [(rng.uniform(0, 20), rng.uniform(0, 20)) for _ in range(15)]
        theta, trans = -0.03, (0.8, -0.5)
        shifted = [apply_se2(theta, trans, p) for p in field]
        inv_theta = -theta
        c, s = math.cos(inv_theta), math.sin(inv_theta)
        inv_trans = (-(c * trans[0] - s * trans[1]), -(s * trans[0] + c * trans[1]))
        obs_pts = shifted  # obs are in the shifted frame; align back onto field
        th, tr = blind_align(obs_pts, field)
        residual = max(
            math.dist(apply_se2(th, tr, o), f) for o, f in zip(obs_pts, field)
        )
        self.assertLess(residual, 1e-6)
        self.assertAlmostEqual(th, inv_theta, places=6)
        self.assertAlmostEqual(tr[0], inv_trans[0], places=6)
        self.assertAlmostEqual(tr[1], inv_trans[1], places=6)


class TestFormTracklets(unittest.TestCase):
    def test_clusters_split_by_gate_and_class(self):
        rows = [
            obs(0.0, 0.0, 0.0), obs(0.1, 0.1, 0.0),
            obs(0.2, 5.0, 5.0),
            obs(0.3, 0.05, 0.05, class_id="car"),
        ]
        tks = form_tracklets(rows)
        self.assertEqual(len(tks), 3)
        classes = sorted(t.class_id for t in tks)
        self.assertEqual(classes, ["car", "mine", "mine"])

    def test_median_trim_rejects_outlier(self):
        rows = [obs(0.1 * i, 0.01 * i, 0.0) for i in range(10)]
        rows.append(obs(2.0, 0.9, 0.0))  # inside gate, far from the body
        tks = form_tracklets(rows)
        self.assertEqual(len(tks), 1)
        self.assertLess(tks[0].center[0], 0.1)

    def test_tags_collected(self):
        rows = [obs(0.0, 0.0, 0.0), obs(0.1, 0.0, 0.0, tag_id="t4")]
        tks = form_tracklets(rows)
        self.assertEqual(tks[0].tag_ids, ["t4"])


class TestAssociator(unittest.TestCase):
    FIELD = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0), (6.0, 6.0), (3.0, 9.0)]

    def _first_pass(self, assoc):
        return assoc.ingest_pass([tracklet(n, e) for n, e in self.FIELD])

    def test_first_pass_births_all(self):
        assoc = Associator()
        res = self._first_pass(assoc)
        self.assertEqual(len(res.born), 5)
        self.assertEqual(res.matches, [])

    def test_second_pass_with_common_mode_error_matches_all(self):
        """Offset + yaw beyond the gate but within the coarse gate: raw
        gating would split every tracklet (A's failure mode); SE(2)-before-
        gate recovers all five."""
        assoc = Associator()
        self._first_pass(assoc)
        theta, trans = 0.02, (0.9, -0.6)
        rng = random.Random(3)
        second = [
            tracklet(*apply_se2(theta, trans, (n + rng.gauss(0, 0.05), e + rng.gauss(0, 0.05))))
            for n, e in self.FIELD
        ]
        res = assoc.ingest_pass(second)
        self.assertEqual(len(res.matches), 5)
        self.assertEqual(res.born, [])
        self.assertEqual(res.missed, [])
        for track in assoc.tracks:
            self.assertEqual(track.n_passes_hit, 2)

    def test_new_object_births_once(self):
        assoc = Associator()
        self._first_pass(assoc)
        second = [tracklet(n, e) for n, e in self.FIELD] + [tracklet(12.0, 12.0)]
        res = assoc.ingest_pass(second)
        self.assertEqual(len(res.matches), 5)
        self.assertEqual(len(res.born), 1)

    def test_one_to_one_prevents_duplicate_merge(self):
        """Two tracklets near one track: exactly one matches, the other
        births (B's duplicate-merge removal)."""
        assoc = Associator()
        assoc.ingest_pass([tracklet(0.0, 0.0)])
        res = assoc.ingest_pass([tracklet(0.05, 0.0), tracklet(0.4, 0.0)])
        self.assertEqual(len(res.matches), 1)
        self.assertEqual(len(res.born), 1)

    def test_class_is_a_hard_gate(self):
        assoc = Associator()
        assoc.ingest_pass([tracklet(0.0, 0.0, class_id="mine")])
        res = assoc.ingest_pass([tracklet(0.0, 0.0, class_id="car")])
        self.assertEqual(res.matches, [])
        self.assertEqual(len(res.born), 1)

    def test_visibility_splits_miss_from_out_of_view(self):
        assoc = Associator()
        self._first_pass(assoc)
        looked = lambda ne: "observed" if ne[0] < 3.0 else "unseen"
        res = assoc.ingest_pass([tracklet(0.0, 0.0)], visibility=looked)
        matched = {t for t, _ in res.matches}
        self.assertEqual(len(matched), 1)
        # in-footprint unmatched tracks are misses; the rest out-of-view
        self.assertEqual(sorted(res.missed), [2])  # (0,6): n<3, observed, unmatched
        self.assertEqual(sorted(res.out_of_view), [1, 3, 4])
        self.assertEqual(assoc.tracks[2].n_passes_missed, 1)
        self.assertEqual(assoc.tracks[1].n_passes_missed, 0)

    def test_out_of_view_track_concedes_to_birth_boundary(self):
        """A tracklet exactly between gate/2 and gate from a track: an
        observed track wins it; an unseen track loses it to birth."""
        assoc = Associator()
        assoc.ingest_pass([tracklet(0.0, 0.0)])
        d = 0.9  # d^2 = 0.81 > birth+miss(unseen)=0.5, < birth+miss(obs)=1.0
        res_obs = assoc.ingest_pass([tracklet(d, 0.0)], visibility=lambda ne: "observed")
        self.assertEqual(len(res_obs.matches), 1)
        assoc2 = Associator()
        assoc2.ingest_pass([tracklet(0.0, 0.0)])
        res_unseen = assoc2.ingest_pass([tracklet(d, 0.0)], visibility=lambda ne: "unseen")
        self.assertEqual(res_unseen.matches, [])
        self.assertEqual(len(res_unseen.born), 1)

    def test_track_center_is_across_pass_mean(self):
        assoc = Associator()
        assoc.ingest_pass([tracklet(0.0, 0.0)])
        assoc.ingest_pass([tracklet(0.2, 0.0)])
        self.assertAlmostEqual(assoc.tracks[0].center[0], 0.1, places=9)


if __name__ == "__main__":
    unittest.main()
