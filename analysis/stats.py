import cv2
import numpy as np
import pandas as pd

from utils import image_to_court, get_foot_position, measure_distance
from trackers.player_tracker import FAR_PLAYER_ID, NEAR_PLAYER_ID

PLAYER_IDS = (FAR_PLAYER_ID, NEAR_PLAYER_ID)


def player_court_positions(player_detections, H, smooth_window=7, max_gap=15):
    """Smoothed foot positions in court meters: list (per frame) of {player_id: (x, y)}."""
    n = len(player_detections)
    out = [dict() for _ in range(n)]
    for pid in PLAYER_IDS:
        rows = [image_to_court(get_foot_position(d[pid]), H) if pid in d else (np.nan, np.nan)
                for d in player_detections]
        df = pd.DataFrame(rows, columns=['x', 'y'])
        df = df.interpolate(limit=max_gap, limit_area='inside')
        df = df.rolling(smooth_window, center=True, min_periods=1).median()
        for i, (x, y) in enumerate(df.to_numpy()):
            if not np.isnan(x):
                out[i][pid] = (float(x), float(y))
    return out


def compute_shot_speeds(hits, bounces, H, fps):
    """Average ball speed (km/h) from each hit to where it lands (or the next hit)."""
    bounce_by_shot = {b['shot']: b for b in bounces}
    shots = []
    for k, hit in enumerate(hits):
        start = image_to_court(get_foot_position(hit['bbox']), H)
        if k in bounce_by_shot:
            end_frame, end = bounce_by_shot[k]['frame'], bounce_by_shot[k]['position']
        elif k + 1 < len(hits):
            end_frame = hits[k + 1]['frame']
            end = image_to_court(get_foot_position(hits[k + 1]['bbox']), H)
        else:
            continue
        dt = (end_frame - hit['frame']) / fps
        if dt <= 0:
            continue
        shots.append({'frame': hit['frame'], 'player_id': hit['player_id'],
                      'speed_kmh': measure_distance(start, end) / dt * 3.6})
    return shots


def compute_frame_stats(n_frames, player_positions, shots, fps, speed_window=15):
    """Per-frame statistics for the overlay."""
    speeds, distances = {}, {}
    for pid in PLAYER_IDS:
        pos = pd.DataFrame([p.get(pid, (np.nan, np.nan)) for p in player_positions], columns=['x', 'y'])
        step = np.sqrt(pos['x'].diff() ** 2 + pos['y'].diff() ** 2)
        speeds[pid] = (step * fps * 3.6).rolling(speed_window, min_periods=1).mean().fillna(0).to_numpy()
        distances[pid] = step.fillna(0).cumsum().to_numpy()

    stats = []
    shot_idx = 0
    done = {pid: [] for pid in PLAYER_IDS}
    for i in range(n_frames):
        while shot_idx < len(shots) and shots[shot_idx]['frame'] <= i:
            done[shots[shot_idx]['player_id']].append(shots[shot_idx]['speed_kmh'])
            shot_idx += 1
        stats.append({pid: {
            'shots': len(done[pid]),
            'last_shot_kmh': done[pid][-1] if done[pid] else None,
            'avg_shot_kmh': float(np.mean(done[pid])) if done[pid] else None,
            'speed_kmh': float(speeds[pid][i]),
            'distance_m': float(distances[pid][i]),
        } for pid in PLAYER_IDS})
    return stats


def draw_stats(frames, stats, origin=(30, 120)):
    def fmt(v, unit):
        return "-" if v is None else f"{v:.0f} {unit}"

    x0, y0 = origin
    w, row_h = 400, 34
    for frame, s in zip(frames, stats):
        p1, p2 = s[FAR_PLAYER_ID], s[NEAR_PLAYER_ID]
        rows = [
            ("", "Player 1", "Player 2"),
            ("Shots", str(p1['shots']), str(p2['shots'])),
            ("Last shot", fmt(p1['last_shot_kmh'], "km/h"), fmt(p2['last_shot_kmh'], "km/h")),
            ("Avg shot", fmt(p1['avg_shot_kmh'], "km/h"), fmt(p2['avg_shot_kmh'], "km/h")),
            ("Speed", fmt(p1['speed_kmh'], "km/h"), fmt(p2['speed_kmh'], "km/h")),
            ("Distance", fmt(p1['distance_m'], "m"), fmt(p2['distance_m'], "m")),
        ]
        h = row_h * len(rows) + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
        frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        for r, (label, a, b) in enumerate(rows):
            y = y0 + 30 + r * row_h
            cv2.putText(frame, label, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, a, (x0 + 140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 255), 1)
            cv2.putText(frame, b, (x0 + 275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 120), 1)
    return frames
