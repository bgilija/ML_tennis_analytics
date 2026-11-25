# Tennis analysis

Tennis video analysis (static broadcast camera): player and ball detection, court keypoints,
mini map, hits, bounces and statistics (shot speed, player speed and distance covered).

## Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Models (not in git):

| file | description |
|---|---|
| `models/yolo5_last.pt` | YOLOv5 ball detector (`training/tennis_ball_detector_training.ipynb`) |
| `models/keypoints_model.pth` | ResNet50, 14 court keypoints (`training/tennis_court_keypoints_training.ipynb`) |
| `yolov8x.pt` | COCO YOLOv8x for players, downloaded automatically by ultralytics |

## Usage

```
python main.py --input input_videos/input_video.avi
python main.py --input input_videos/ex04.avi --use-stubs   # reuse cached detections
```

Output: `output_videos/<name>_output.avi`. Detections are cached in `tracker_stubs/<name>_*.pkl`.
The input must be an `.avi` file.

## Pipeline

1. `CourtLineDetector.predict_video` – 14 court keypoints (median over several frames).
2. `utils/court_reference.py` – homography from image to court meters (all 14 keypoints, RANSAC).
3. `PlayerTracker` – YOLOv8 + tracking; two players with stable IDs: 1 = far, 2 = near.
4. `BallTracker` – candidate selection by trajectory continuity, outlier removal, interpolation of short gaps.
5. `analysis/events.py` – hits (ball changes direction next to a player) and bounces (spike in y acceleration).
6. `analysis/stats.py` – shot speeds, player speed and distance covered, statistics panel.
7. `MiniCourt` – mini map in meters.

## Tests

```
python -m unittest discover tests
```

The tests do not use the models (synthetic data): homography, ball selection, outliers,
interpolation, player selection.

## Known limitations

- The camera must be static (court keypoints are computed once for the whole video).
- A bounce is detected as a spike in the ball trajectory's acceleration; if the model misses
  the ball right around the bounce, its position can be off by a few frames.
- On the far player's serve, the top of the ball toss can be detected as a hit.
- Shot speed is the average speed from the player to the bounce, not the speed off the racket.
