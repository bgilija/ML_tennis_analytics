from ultralytics import YOLO
import cv2
import pickle
import os

from utils import (
    measure_distance,
    get_foot_position,
    image_to_court,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
)

FAR_PLAYER_ID = 1
NEAR_PLAYER_ID = 2

# Area around the court (meters) where a player can realistically stand.
_MARGIN_X_M = 4.0
_MARGIN_Y_M = 7.0


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # ---------------------------------------------------------------- detection
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)[0]
        player_dict = {}
        for box in results.boxes:
            if box.id is None:  # tracker did not (yet) assign an id
                continue
            track_id = int(box.id.item())
            player_dict[track_id] = box.xyxy.tolist()[0]
        return player_dict

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        player_detections = [self.detect_frame(frame) for frame in frames]

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    # ---------------------------------------------------------------- selection
    @staticmethod
    def _distance_to_half(pos_m, far_side):
        """0 inside that half of the court, otherwise distance (m) to it."""
        x, y = pos_m
        y_min, y_max = (0.0, NET_Y_M) if far_side else (NET_Y_M, COURT_LENGTH_M)
        dx = max(0.0 - x, 0.0, x - COURT_WIDTH_M)
        dy = max(y_min - y, 0.0, y - y_max)
        return (dx ** 2 + dy ** 2) ** 0.5

    @staticmethod
    def _in_play_area(pos_m):
        x, y = pos_m
        return (-_MARGIN_X_M <= x <= COURT_WIDTH_M + _MARGIN_X_M and
                -_MARGIN_Y_M <= y <= COURT_LENGTH_M + _MARGIN_Y_M)

    def choose_and_filter_players(self, H, player_detections, max_speed_m_per_frame=0.6):
        """Keep exactly the two players, with stable ids 1 (far) and 2 (near).

        Each frame, every slot keeps following its tracker id. If that id is lost
        (occlusion / id switch), the slot is re-assigned to the person on the same
        side of the net closest to where the player was last seen.
        """
        slots = {FAR_PLAYER_ID: {'far': True, 'track_id': None, 'pos': None, 'last_frame': None},
                 NEAR_PLAYER_ID: {'far': False, 'track_id': None, 'pos': None, 'last_frame': None}}
        filtered = []

        for i, detections in enumerate(player_detections):
            positions = {tid: image_to_court(get_foot_position(bb), H) for tid, bb in detections.items()}
            positions = {tid: p for tid, p in positions.items() if self._in_play_area(p)}
            frame_result = {}
            used = set()

            for pid, slot in slots.items():
                tid = slot['track_id']
                if tid not in positions or tid in used:
                    tid = None
                    candidates = [t for t, p in positions.items()
                                  if t not in used and (p[1] < NET_Y_M) == slot['far']]
                    if candidates:
                        if slot['pos'] is None:
                            # first time: closest to that half of the court, then to the centre line
                            tid = min(candidates, key=lambda t: (
                                round(self._distance_to_half(positions[t], slot['far']), 1),
                                abs(positions[t][0] - COURT_WIDTH_M / 2)))
                        else:
                            gap = i - slot['last_frame']
                            best = min(candidates, key=lambda t: measure_distance(positions[t], slot['pos']))
                            if measure_distance(positions[best], slot['pos']) <= 2.0 + max_speed_m_per_frame * gap:
                                tid = best
                if tid is not None:
                    slot.update(track_id=tid, pos=positions[tid], last_frame=i)
                    frame_result[pid] = detections[tid]
                    used.add(tid)
            filtered.append(frame_result)

        return filtered

    # ---------------------------------------------------------------- drawing
    def draw_boxes(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for player_id, bounding_box in player_dict.items():
                x1, y1, x2, y2 = map(int, bounding_box)
                cv2.putText(frame, f"Player {player_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            output_frames.append(frame)
        return output_frames
