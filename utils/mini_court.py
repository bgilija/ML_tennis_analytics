import cv2
import numpy as np
from utils.bounding_box_utils import get_center_of_bounding_box


class MiniCourt:
    CANVAS_WIDTH = 300
    CANVAS_HEIGHT = 560
    BUFFER = 50
    COURT_PADDING = 30

    SCALE = 21.04           # px/m
    COURT_LENGTH_M = 23.77
    COURT_WIDTH_M = 10.97
    SINGLES_ALLEY_M = 1.37
    SERVICE_DEPTH_M = 6.40

    def __init__(self, frame):
        self._set_canvas_position(frame)
        self._set_court_position()
        self.court_keypoints = self._build_court_keypoints()

    def _meters_to_px(self, meters):
        return int(meters * self.SCALE)

    def _set_canvas_position(self, frame):
        h, w = frame.shape[:2]
        self.canvas_x = w - self.CANVAS_WIDTH - self.BUFFER
        self.canvas_y = h - self.CANVAS_HEIGHT - self.BUFFER

    def _set_court_position(self):
        self.court_x = self.canvas_x + self.COURT_PADDING
        self.court_y = self.canvas_y + self.COURT_PADDING
        self.court_w = self._meters_to_px(self.COURT_WIDTH_M)
        self.court_h = self._meters_to_px(self.COURT_LENGTH_M)

    def _build_court_keypoints(self):
        x0 = self.court_x
        y0 = self.court_y
        w = self.court_w
        h = self.court_h
        sl = self._meters_to_px(self.SINGLES_ALLEY_M)
        svc = self._meters_to_px(self.SERVICE_DEPTH_M)
        cx = x0 + w // 2
        cy = y0 + h // 2

        # 14 keypoints matching CourtLineDetector output order:
        # 0-3: court corners (far-left, far-right, near-right, near-left)
        # 4-7: service line × singles sideline intersections
        # 8-9: service T points
        # 10-11: net × singles sideline
        # 12-13: baseline center marks
        return [
            (x0,          y0),              # 0: far baseline, left doubles
            (x0 + w,      y0),              # 1: far baseline, right doubles
            (x0 + w,      y0 + h),          # 2: near baseline, right doubles
            (x0,          y0 + h),          # 3: near baseline, left doubles
            (x0 + sl,     y0 + svc),        # 4: far service × left singles
            (x0 + w - sl, y0 + svc),        # 5: far service × right singles
            (x0 + w - sl, y0 + h - svc),    # 6: near service × right singles
            (x0 + sl,     y0 + h - svc),    # 7: near service × left singles
            (cx,          y0 + svc),        # 8: far service T
            (cx,          y0 + h - svc),    # 9: near service T
            (x0 + sl,     cy),              # 10: net × left singles
            (x0 + w - sl, cy),              # 11: net × right singles
            (cx,          y0),              # 12: far baseline center
            (cx,          y0 + h),          # 13: near baseline center
        ]

    def compute_homography(self, court_keypoints_flat):
        pts = np.array(
            [[court_keypoints_flat[i * 2], court_keypoints_flat[i * 2 + 1]] for i in range(14)],
            dtype=np.float32,
        )
        # Identify 4 court corners geometrically — independent of model keypoint ordering.
        # For a standard broadcast end-view: far baseline is at top (small y), near at bottom (large y).
        # min(x+y) → top-left, max(x-y) → top-right, max(x+y) → bot-right, min(x-y) → bot-left
        sums  = pts[:, 0] + pts[:, 1]
        diffs = pts[:, 0] - pts[:, 1]
        far_left   = pts[np.argmin(sums)]
        near_right = pts[np.argmax(sums)]
        far_right  = pts[np.argmax(diffs)]
        near_left  = pts[np.argmin(diffs)]

        src = np.array([far_left, far_right, near_right, near_left], dtype=np.float32)
        kp  = self.court_keypoints
        dst = np.array([kp[0], kp[1], kp[2], kp[3]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def transform_point(self, point, H):
        pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        return (int(result[0][0][0]), int(result[0][0][1]))

    @staticmethod
    def _foot_position(bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, H):
        player_mini, ball_mini = [], []
        for frame_players, frame_ball in zip(player_boxes, ball_boxes):
            player_mini.append({
                pid: self.transform_point(self._foot_position(bbox), H)
                for pid, bbox in frame_players.items()
            })
            ball_mini.append({
                bid: self.transform_point(get_center_of_bounding_box(bbox), H)
                for bid, bbox in frame_ball.items()
            })
        return player_mini, ball_mini

    def draw_mini_court(self, frames):
        output = []
        for frame in frames:
            frame = self._draw_background(frame)
            frame = self._draw_court(frame)
            output.append(frame)
        return output

    def draw_player_ball_positions(self, frames, player_mini_positions, ball_mini_positions):
        player_colors = {}
        palette = [(0, 0, 255), (255, 0, 0)]

        output = []
        for frame, frame_players, frame_ball in zip(frames, player_mini_positions, ball_mini_positions):
            for pid, pos in frame_players.items():
                if pid not in player_colors:
                    player_colors[pid] = palette[len(player_colors) % len(palette)]
                cv2.circle(frame, pos, 5, player_colors[pid], -1)

            for pos in frame_ball.values():
                cv2.circle(frame, pos, 5, (0, 255, 255), -1)

            output.append(frame)
        return output

    def _draw_background(self, frame):
        overlay = frame.copy()
        cx, cy = self.canvas_x, self.canvas_y
        cv2.rectangle(overlay, (cx, cy), (cx + self.CANVAS_WIDTH, cy + self.CANVAS_HEIGHT), (40, 40, 40), -1)
        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    def _draw_court(self, frame):
        kp = self.court_keypoints
        x0, y0 = self.court_x, self.court_y
        w, h = self.court_w, self.court_h
        sl = self._meters_to_px(self.SINGLES_ALLEY_M)
        cy = y0 + h // 2

        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (34, 139, 34), -1)

        lines = [
            (kp[0], kp[1]),
            (kp[3], kp[2]),
            (kp[0], kp[3]),
            (kp[1], kp[2]),
            ((x0 + sl, y0),      (x0 + sl, y0 + h)),
            ((x0 + w - sl, y0),  (x0 + w - sl, y0 + h)),
            (kp[4], kp[5]),
            (kp[7], kp[6]),
            (kp[8], kp[9]),
            ((x0, cy), (x0 + w, cy)),
        ]
        for p1, p2 in lines:
            cv2.line(frame, p1, p2, (255, 255, 255), 1)

        return frame
