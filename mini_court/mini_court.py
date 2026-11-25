import cv2

from utils import COURT_KEYPOINTS_M, COURT_LINES, COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M


class MiniCourt:
    """Top-down court drawn in the bottom-right corner. Works in court meters
    (see utils/court_reference.py) and converts them to canvas pixels."""

    CANVAS_HEIGHT = 560
    BUFFER = 50
    # Area shown around the court, so players behind the baseline are not
    # squashed onto it.
    MARGIN_X_M = 2.5
    MARGIN_Y_M = 5.0

    PLAYER_COLORS = {1: (0, 0, 255), 2: (255, 0, 0)}
    BALL_COLOR = (0, 255, 255)
    BOUNCE_COLOR = (0, 255, 0)

    def __init__(self, frame):
        self.scale = self.CANVAS_HEIGHT / (COURT_LENGTH_M + 2 * self.MARGIN_Y_M)  # px per meter
        self.canvas_w = int(round((COURT_WIDTH_M + 2 * self.MARGIN_X_M) * self.scale))
        self.canvas_h = self.CANVAS_HEIGHT

        h, w = frame.shape[:2]
        self.canvas_x = w - self.canvas_w - self.BUFFER
        self.canvas_y = h - self.canvas_h - self.BUFFER

    # ------------------------------------------------------------ coordinates
    def court_to_mini(self, point_m):
        x_m = min(max(point_m[0], -self.MARGIN_X_M), COURT_WIDTH_M + self.MARGIN_X_M)
        y_m = min(max(point_m[1], -self.MARGIN_Y_M), COURT_LENGTH_M + self.MARGIN_Y_M)
        x = self.canvas_x + (x_m + self.MARGIN_X_M) * self.scale
        y = self.canvas_y + (y_m + self.MARGIN_Y_M) * self.scale
        return int(round(x)), int(round(y))

    # ------------------------------------------------------------ drawing
    def draw_mini_court(self, frames):
        return [self._draw_court(self._draw_background(frame)) for frame in frames]

    def _draw_background(self, frame):
        overlay = frame.copy()
        cx, cy = self.canvas_x, self.canvas_y
        cv2.rectangle(overlay, (cx, cy), (cx + self.canvas_w, cy + self.canvas_h), (40, 40, 40), -1)
        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    def _draw_court(self, frame):
        tl = self.court_to_mini((0, 0))
        br = self.court_to_mini((COURT_WIDTH_M, COURT_LENGTH_M))
        cv2.rectangle(frame, tl, br, (34, 139, 34), -1)
        for a, b in COURT_LINES:
            cv2.line(frame, self.court_to_mini(COURT_KEYPOINTS_M[a]), self.court_to_mini(COURT_KEYPOINTS_M[b]),
                     (255, 255, 255), 1)
        # net
        cv2.line(frame, self.court_to_mini((-0.9, NET_Y_M)), self.court_to_mini((COURT_WIDTH_M + 0.9, NET_Y_M)),
                 (200, 200, 200), 2)
        return frame

    def draw_player_ball_positions(self, frames, player_positions_m, ball_positions_m):
        for frame, players, ball in zip(frames, player_positions_m, ball_positions_m):
            for pid, pos in players.items():
                cv2.circle(frame, self.court_to_mini(pos), 8, self.PLAYER_COLORS.get(pid, (255, 255, 255)), -1)
            if ball is not None:
                cv2.circle(frame, self.court_to_mini(ball), 5, self.BALL_COLOR, -1)
        return frames

    def draw_bounces(self, frames, bounces):
        for i, frame in enumerate(frames):
            for b in bounces:
                if b['frame'] <= i:
                    color = self.BOUNCE_COLOR if b['in'] else (0, 0, 255)
                    cv2.circle(frame, self.court_to_mini(b['position']), 4, color, -1)
        return frames
