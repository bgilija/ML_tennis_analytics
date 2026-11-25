from ultralytics import YOLO
import cv2
import pickle
import os
import numpy as np
import pandas as pd

from utils import measure_distance, get_center_of_bounding_box


class BallTracker:
    def __init__(self, model_path, conf=0.15):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_frame(self, frame):
        # Keep every candidate with its confidence; the choice is made later
        # using trajectory continuity (see select_ball_candidates).
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            candidates.append([x1, y1, x2, y2, float(box.conf.item())])
        return candidates

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        ball_candidates = [self.detect_frame(frame) for frame in frames]

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(ball_candidates, f)

        return ball_candidates

    @staticmethod
    def remove_static_candidates(ball_candidates, radius_px=4, min_frames=8):
        """Drop detections that sit on the same spot in many frames (logos, a ball
        lying on the ground, ...). A ball in play does not stay still that long."""
        centers = [get_center_of_bounding_box(c) for cs in ball_candidates for c in cs]
        if not centers:
            return ball_candidates
        grid = {}
        for x, y in centers:
            key = (x // radius_px, y // radius_px)
            grid[key] = grid.get(key, 0) + 1

        def count_near(x, y):
            kx, ky = x // radius_px, y // radius_px
            return sum(grid.get((kx + dx, ky + dy), 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))

        return [[c for c in cs if count_near(*get_center_of_bounding_box(c)) < min_frames]
                for cs in ball_candidates]

    def select_ball_candidates(self, ball_candidates, max_jump_px=120, max_gap=10):
        """One detection per frame. While the ball was seen recently only a
        candidate within reach of its last position is accepted (otherwise the
        frame is left empty for interpolation); after a longer gap the most
        confident candidate re-acquires the ball."""
        ball_candidates = self.remove_static_candidates(ball_candidates)
        selected = []
        last_center, last_frame = None, None
        for i, candidates in enumerate(ball_candidates):
            chosen = None
            if candidates:
                if last_center is not None and i - last_frame <= max_gap:
                    limit = max_jump_px * (i - last_frame)
                    dists = [measure_distance(get_center_of_bounding_box(c), last_center) for c in candidates]
                    best = int(np.argmin(dists))
                    if dists[best] <= limit:
                        chosen = candidates[best]
                else:
                    chosen = max(candidates, key=lambda c: c[4])
            if chosen is None:
                selected.append({})
                continue
            selected.append({1: chosen[:4]})
            last_center, last_frame = get_center_of_bounding_box(chosen), i
        return selected

    def remove_outliers(self, ball_detections, max_jump_px=120):
        """Drop isolated detections that jump far from both neighbouring detections."""
        frames_with_ball = [i for i, d in enumerate(ball_detections) if d]
        cleaned = [dict(d) for d in ball_detections]
        for k, i in enumerate(frames_with_ball):
            center = get_center_of_bounding_box(ball_detections[i][1])
            neighbour_ok = False
            has_neighbour = False
            for j in (frames_with_ball[k - 1] if k > 0 else None,
                      frames_with_ball[k + 1] if k + 1 < len(frames_with_ball) else None):
                if j is None:
                    continue
                has_neighbour = True
                gap = abs(i - j)
                if measure_distance(center, get_center_of_bounding_box(ball_detections[j][1])) <= max_jump_px * gap:
                    neighbour_ok = True
            if has_neighbour and not neighbour_ok:
                cleaned[i] = {}
        return cleaned

    def interpolate_ball_positions(self, ball_positions, max_gap=20):
        """Linear interpolation over gaps of at most `max_gap` frames.
        Longer gaps (ball out of view) stay empty instead of being invented."""
        rows = [x.get(1, [np.nan] * 4) for x in ball_positions]
        df = pd.DataFrame(rows, columns=['x1', 'y1', 'x2', 'y2'])
        if df.isna().all().all():
            return [{} for _ in ball_positions]

        missing = df['x1'].isna()
        gap_id = (missing != missing.shift()).cumsum()
        gap_len = missing.groupby(gap_id).transform('sum')
        df = df.interpolate(limit_area='inside')
        # leave long gaps empty
        df.loc[missing & (gap_len > max_gap)] = np.nan

        return [{} if np.isnan(r).any() else {1: r.tolist()} for r in df.to_numpy()]

    def draw_boxes(self, frames, ball_detections):
        output_frames = []
        for frame, ball_dict in zip(frames, ball_detections):
            for track_id, bounding_box in ball_dict.items():
                x1, y1, x2, y2 = map(int, bounding_box[:4])
                cv2.putText(frame, "Ball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            output_frames.append(frame)
        return output_frames
