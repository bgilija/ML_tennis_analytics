"""Model-free tests. Run with:  python -m unittest discover tests"""
import unittest

import cv2
import numpy as np

from utils import COURT_KEYPOINTS_M, compute_court_homography, image_to_court, court_to_image
from trackers import BallTracker, PlayerTracker
from analysis import build_ball_ground_track


def _fake_image_keypoints():
    # Project the court model into a fake "broadcast" view.
    dst = np.float32([[600, 300], [1320, 300], [1590, 870], [340, 870]])
    src = COURT_KEYPOINTS_M[[0, 1, 3, 2]]
    H_court_to_img = cv2.getPerspectiveTransform(src, dst)
    return cv2.perspectiveTransform(COURT_KEYPOINTS_M.reshape(-1, 1, 2), H_court_to_img).reshape(-1)


def _box(cx, cy, size=10):
    return [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]


class TestHomography(unittest.TestCase):
    def test_round_trip_and_keypoints(self):
        kp = _fake_image_keypoints()
        H, inliers = compute_court_homography(kp)
        self.assertEqual(inliers.sum(), 14)
        for i, (x, y) in enumerate(kp.reshape(-1, 2)):
            mx, my = image_to_court((x, y), H)
            self.assertAlmostEqual(mx, COURT_KEYPOINTS_M[i][0], places=2)
            self.assertAlmostEqual(my, COURT_KEYPOINTS_M[i][1], places=2)
        ix, iy = court_to_image((3.0, 7.0), H)
        mx, my = image_to_court((ix, iy), H)
        self.assertAlmostEqual(mx, 3.0, places=3)
        self.assertAlmostEqual(my, 7.0, places=3)

    def test_one_bad_keypoint_is_rejected(self):
        kp = _fake_image_keypoints().copy()
        kp[2 * 13] += 150  # keypoint 13 badly off
        H, inliers = compute_court_homography(kp)
        self.assertFalse(inliers[13])
        mx, my = image_to_court(kp.reshape(-1, 2)[0], H)
        self.assertAlmostEqual(mx, 0.0, places=1)
        self.assertAlmostEqual(my, 0.0, places=1)


class TestBallPostprocessing(unittest.TestCase):
    def setUp(self):
        self.bt = BallTracker.__new__(BallTracker)  # no model needed

    def test_select_prefers_continuity_over_confidence(self):
        cands = [[_box(100, 100) + [0.9]],
                 [_box(110, 100) + [0.4], _box(900, 500) + [0.95]]]
        sel = self.bt.select_ball_candidates(cands)
        self.assertAlmostEqual((sel[1][1][0] + sel[1][1][2]) / 2, 110)

    def test_select_first_frame_uses_confidence(self):
        sel = self.bt.select_ball_candidates([[_box(100, 100) + [0.3], _box(500, 500) + [0.8]]])
        self.assertAlmostEqual((sel[0][1][0] + sel[0][1][2]) / 2, 500)

    def test_static_false_positive_is_ignored(self):
        # a logo detected at the same spot in every frame, the real ball moving
        cands = [[_box(100 + 10 * i, 100) + [0.6], _box(1467, 836) + [0.25]] for i in range(10)]
        cands[5] = [_box(1467, 836) + [0.25]]  # ball missed, only the logo left
        sel = self.bt.select_ball_candidates(cands)
        self.assertEqual(sel[5], {})
        self.assertAlmostEqual((sel[6][1][0] + sel[6][1][2]) / 2, 160)

    def test_no_jump_to_far_candidate_while_tracking(self):
        cands = [[_box(100, 100) + [0.6]], [_box(900, 800) + [0.9]], [_box(120, 100) + [0.6]]]
        sel = self.bt.select_ball_candidates(cands)
        self.assertEqual(sel[1], {})
        self.assertTrue(sel[2])

    def test_outlier_removed(self):
        dets = [{1: _box(100 + 10 * i, 100)} for i in range(5)]
        dets[2] = {1: _box(900, 900)}
        cleaned = self.bt.remove_outliers(dets)
        self.assertEqual(cleaned[2], {})
        self.assertTrue(all(cleaned[i] for i in (0, 1, 3, 4)))

    def test_interpolation_short_and_long_gaps(self):
        dets = [{1: _box(0, 0)}, {}, {}, {1: _box(30, 0)}] + [{}] * 25 + [{1: _box(0, 0)}]
        out = self.bt.interpolate_ball_positions(dets, max_gap=20)
        self.assertAlmostEqual((out[1][1][0] + out[1][1][2]) / 2, 10)
        self.assertTrue(all(out[i] == {} for i in range(4, 29)))
        self.assertTrue(out[29])

    def test_interpolation_no_ball_at_all(self):
        self.assertEqual(self.bt.interpolate_ball_positions([{}, {}]), [{}, {}])

    def test_leading_and_trailing_gaps_stay_empty(self):
        out = self.bt.interpolate_ball_positions([{}, {1: _box(5, 5)}, {}])
        self.assertEqual(out[0], {})
        self.assertEqual(out[2], {})


class TestPlayerSelection(unittest.TestCase):
    def setUp(self):
        self.pt = PlayerTracker.__new__(PlayerTracker)
        self.kp = _fake_image_keypoints()
        self.H, _ = compute_court_homography(self.kp)
        H_inv = np.linalg.inv(self.H)

        def person_at(mx, my, h=120):
            x, y = cv2.perspectiveTransform(np.float32([[[mx, my]]]), H_inv)[0][0]
            return [x - 20, y - h, x + 20, y]
        self.person_at = person_at

    def test_picks_players_and_ignores_bystanders(self):
        frame = {
            10: self.person_at(5.5, -1.0),     # far player behind baseline
            11: self.person_at(5.0, 25.0),     # near player behind baseline
            12: self.person_at(12.5, 11.0),    # umpire next to the net
            13: self.person_at(-3.5, -5.0),    # ball kid in the corner
        }
        out = self.pt.choose_and_filter_players(self.H, [frame])
        self.assertEqual(out[0][1], frame[10])
        self.assertEqual(out[0][2], frame[11])

    def test_recovers_after_track_id_switch(self):
        f0 = {1: self.person_at(5.5, -1.0), 2: self.person_at(5.0, 25.0)}
        f1 = {1: self.person_at(5.6, -1.0), 7: self.person_at(5.1, 25.0)}  # near player got new id 7
        out = self.pt.choose_and_filter_players(self.H, [f0, f1])
        self.assertEqual(out[1][2], f1[7])
        self.assertEqual(out[1][1], f1[1])

    def test_missing_box_id_does_not_crash_selection(self):
        out = self.pt.choose_and_filter_players(self.H, [{}, {}])
        self.assertEqual(out, [{}, {}])


class TestBallGroundTrack(unittest.TestCase):
    def test_interpolates_between_hit_and_bounce(self):
        H = np.eye(3)
        hits = [{'frame': 0, 'player_id': 2, 'bbox': [0, 0, 10, 20]}]  # foot (5, 20)
        bounces = [{'frame': 10, 'position': (5.0, 0.0), 'in': True, 'shot': 0}]
        track = build_ball_ground_track(12, hits, bounces, H)
        self.assertAlmostEqual(track[5][1], 10.0, places=3)
        self.assertIsNone(track[11])


if __name__ == "__main__":
    unittest.main()
