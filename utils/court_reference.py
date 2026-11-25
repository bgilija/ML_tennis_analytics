"""Real-world tennis court geometry and the image -> court (meters) homography.

Court coordinate system (meters):
    origin = far baseline, left doubles corner (as seen from the broadcast camera)
    x grows to the right (0 .. 10.97), y grows towards the camera (0 .. 23.77)
"""
import cv2
import numpy as np

COURT_LENGTH_M = 23.77
COURT_WIDTH_M = 10.97
SINGLES_ALLEY_M = 1.37
NET_TO_SERVICE_LINE_M = 6.40
NET_Y_M = COURT_LENGTH_M / 2

_SL = SINGLES_ALLEY_M
_SW = COURT_WIDTH_M - SINGLES_ALLEY_M
_CX = COURT_WIDTH_M / 2
_FAR_SVC = NET_Y_M - NET_TO_SERVICE_LINE_M
_NEAR_SVC = NET_Y_M + NET_TO_SERVICE_LINE_M

# Same order as the CourtLineDetector output (verified on real frames).
COURT_KEYPOINTS_M = np.array([
    (0.0, 0.0),                       # 0  far baseline x left doubles line
    (COURT_WIDTH_M, 0.0),             # 1  far baseline x right doubles line
    (0.0, COURT_LENGTH_M),            # 2  near baseline x left doubles line
    (COURT_WIDTH_M, COURT_LENGTH_M),  # 3  near baseline x right doubles line
    (_SL, 0.0),                       # 4  far baseline x left singles line
    (_SL, COURT_LENGTH_M),            # 5  near baseline x left singles line
    (_SW, 0.0),                       # 6  far baseline x right singles line
    (_SW, COURT_LENGTH_M),            # 7  near baseline x right singles line
    (_SL, _FAR_SVC),                  # 8  far service line x left singles line
    (_SW, _FAR_SVC),                  # 9  far service line x right singles line
    (_SL, _NEAR_SVC),                 # 10 near service line x left singles line
    (_SW, _NEAR_SVC),                 # 11 near service line x right singles line
    (_CX, _FAR_SVC),                  # 12 far service T
    (_CX, _NEAR_SVC),                 # 13 near service T
], dtype=np.float32)

# Line segments between keypoints, used for drawing the court.
COURT_LINES = [
    (0, 1), (2, 3),      # baselines
    (0, 2), (1, 3),      # doubles sidelines
    (4, 5), (6, 7),      # singles sidelines
    (8, 9), (10, 11),    # service lines
    (12, 13),            # centre service line
]


def compute_court_homography(image_keypoints_flat, ransac_threshold_px=10.0):
    """Homography that maps image pixels to court meters, fitted on all 14 keypoints."""
    src = np.asarray(image_keypoints_flat, dtype=np.float32).reshape(-1, 2)
    # RANSAC threshold is measured in the *destination* space, so fit court -> image
    # (threshold in pixels) and invert, instead of image -> court (threshold in meters).
    H_court_to_img, inliers = cv2.findHomography(COURT_KEYPOINTS_M, src, cv2.RANSAC, ransac_threshold_px)
    if H_court_to_img is None:
        raise RuntimeError("Could not compute court homography from the detected keypoints")
    H = np.linalg.inv(H_court_to_img)
    return H / H[2, 2], inliers.ravel().astype(bool)


def reprojection_errors_px(image_keypoints_flat, H):
    """Per-keypoint distance (pixels) between detected keypoints and court model projected back to the image."""
    src = np.asarray(image_keypoints_flat, dtype=np.float32).reshape(-1, 2)
    back = cv2.perspectiveTransform(COURT_KEYPOINTS_M.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    return np.linalg.norm(src - back, axis=1)


def image_to_court(point, H):
    pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(pt, H)[0][0]
    return float(x), float(y)


def court_to_image(point_m, H):
    pt = np.array([[[float(point_m[0]), float(point_m[1])]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(pt, np.linalg.inv(H))[0][0]
    return float(x), float(y)
