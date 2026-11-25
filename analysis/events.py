"""Hit and bounce detection from the (image-space) ball trajectory.

Image y grows downwards. Gravity alone gives the ball's image y a positive
acceleration; when the ball bounces its vertical motion flips, which shows up
as a sharp *negative* spike in the second difference of y - for balls flying
towards the camera as well as away from it. Hits are the moments where the
ball reverses its direction along the court (dy changes sign and stays
changed) while it is next to the player whose side it is leaving.
"""
import numpy as np
import pandas as pd

from utils import (
    image_to_court,
    get_center_of_bounding_box,
    get_foot_position,
    NET_Y_M,
    COURT_LENGTH_M,
    COURT_KEYPOINTS_M,
)
from trackers.player_tracker import FAR_PLAYER_ID, NEAR_PLAYER_ID

SINGLES_X_MIN = float(COURT_KEYPOINTS_M[4][0])
SINGLES_X_MAX = float(COURT_KEYPOINTS_M[6][0])


def _ball_center_series(ball_detections, repeat_tolerance_px=0.5):
    """Ball centre x/y per frame. Frames where the ball did not move at all are
    treated as duplicated video frames (common after re-encoding) and replaced by
    interpolation - otherwise the trajectory becomes a staircase and its
    derivatives are useless."""
    xs, ys = [], []
    prev = None
    for d in ball_detections:
        if d:
            x1, y1, x2, y2 = d[1][:4]
            c = ((x1 + x2) / 2, (y1 + y2) / 2)
            if prev is not None and abs(c[0] - prev[0]) <= repeat_tolerance_px and abs(c[1] - prev[1]) <= repeat_tolerance_px:
                xs.append(np.nan)
                ys.append(np.nan)
            else:
                xs.append(c[0])
                ys.append(c[1])
            prev = c
        else:
            xs.append(np.nan)
            ys.append(np.nan)
            prev = None
    x = pd.Series(xs, dtype=float).interpolate(limit=3, limit_area='inside')
    y = pd.Series(ys, dtype=float).interpolate(limit=3, limit_area='inside')
    return x, y


def _distance_to_box_normalised(point, bbox):
    """Distance (px) from point to the bbox (0 inside), divided by bbox height."""
    x1, y1, x2, y2 = bbox
    dx = max(x1 - point[0], 0, point[0] - x2)
    dy = max(y1 - point[1], 0, point[1] - y2)
    return (dx ** 2 + dy ** 2) ** 0.5 / max(y2 - y1, 1)


def detect_hits(ball_detections, player_detections, window=8, min_gap=20, min_motion_px=0.5,
                max_player_distance=0.7):
    _, y = _ball_center_series(ball_detections)
    dy = y.rolling(5, center=True, min_periods=1).mean().diff()
    direction = np.sign(dy.where(dy.abs() >= min_motion_px))  # NaN where unknown / still

    n = len(y)
    candidates = []
    for i in range(window, n - window):
        before = direction.iloc[i - window:i].dropna()
        after = direction.iloc[i:i + window].dropna()
        if len(before) < window // 2 or len(after) < window // 2:
            continue
        s_before = np.sign(before.mean()) if abs(before.mean()) >= 0.5 else 0
        s_after = np.sign(after.mean()) if abs(after.mean()) >= 0.5 else 0
        if s_before != 0 and s_after == -s_before:
            candidates.append((i, s_after))

    # group consecutive candidate frames, take the turning point of each group
    groups = []
    for i, s in candidates:
        if groups and i - groups[-1][-1][0] <= 2 and groups[-1][-1][1] == s:
            groups[-1].append((i, s))
        else:
            groups.append([(i, s)])

    hits = []
    for g in groups:
        frames = [i for i, _ in g]
        s_after = g[0][1]
        seg = y.iloc[frames[0] - 2:frames[-1] + 3]
        turn = int(seg.idxmax() if s_after < 0 else seg.idxmin())
        # moving up the image afterwards (towards the far side) => near player hit it
        hitter = NEAR_PLAYER_ID if s_after < 0 else FAR_PLAYER_ID

        bbox, ball_bb = None, None
        for k in range(turn - 3, turn + 4):  # tolerate a missing box around the turn
            if 0 <= k < n and hitter in player_detections[k] and ball_detections[k]:
                bbox, ball_bb = player_detections[k][hitter], ball_detections[k][1]
                break
        if bbox is None:
            continue
        if _distance_to_box_normalised(get_center_of_bounding_box(ball_bb), bbox) > max_player_distance:
            continue
        if hits and turn - hits[-1]['frame'] < min_gap:
            continue
        hits.append({'frame': turn, 'player_id': hitter, 'bbox': bbox})
    return hits


def detect_bounces(ball_detections, hits, H, edge_frames=3, min_spike_px=2.0, max_flight_frames=90):
    """At most one bounce per shot: the strongest negative spike of d2y between
    a hit and the next hit, that lands on the receiver's side of the net."""
    _, y = _ball_center_series(ball_detections)
    ddy = y.rolling(3, center=True, min_periods=1).mean().diff().diff().shift(-1)
    n = len(y)

    bounces = []
    for k, hit in enumerate(hits):
        start = hit['frame'] + edge_frames
        end = hits[k + 1]['frame'] - edge_frames if k + 1 < len(hits) else min(n - 1, hit['frame'] + max_flight_frames)
        if end - start < 3:
            continue
        seg = ddy.iloc[start:end + 1].dropna()
        receiver_far = hit['player_id'] == NEAR_PLAYER_ID
        for frame in seg.sort_values().index[:5]:  # strongest spikes first
            if seg[frame] > -min_spike_px:
                break
            frame = int(frame)
            if not ball_detections[frame]:
                continue
            x1, y1, x2, y2 = ball_detections[frame][1][:4]
            pos = image_to_court(((x1 + x2) / 2, y2), H)
            if (pos[1] < NET_Y_M) != receiver_far:
                continue
            in_court = (SINGLES_X_MIN - 0.1 <= pos[0] <= SINGLES_X_MAX + 0.1 and
                        -0.1 <= pos[1] <= COURT_LENGTH_M + 0.1)
            bounces.append({'frame': frame, 'position': pos, 'in': in_court, 'shot': k})
            break
    return bounces


def build_ball_ground_track(n_frames, hits, bounces, H, max_gap_frames=90):
    """Ball position on the ground plane (meters) per frame.

    The homography is only valid for points on the ground, so the ball is
    anchored where it touches the ground or a player (bounces, hits) and is
    linearly interpolated in between. Frames outside these anchors get None.
    """
    anchors = [(h['frame'], image_to_court(get_foot_position(h['bbox']), H)) for h in hits]
    anchors += [(b['frame'], b['position']) for b in bounces]
    anchors.sort(key=lambda a: a[0])

    track = [None] * n_frames
    for (f0, p0), (f1, p1) in zip(anchors, anchors[1:]):
        if f1 - f0 > max_gap_frames or f1 == f0:
            continue
        for f in range(f0, f1 + 1):
            t = (f - f0) / (f1 - f0)
            track[f] = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
    if anchors:
        track[anchors[-1][0]] = anchors[-1][1]
    return track
