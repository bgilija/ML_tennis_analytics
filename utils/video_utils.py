import os
import cv2


def save(video_frames, output_path, fps=24):
    if not video_frames:
        return
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in video_frames:
        out.write(frame)
    out.release()


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")
    return frames, fps


def draw_frame_numbers(frames):
    # Must run after all detection, so the text never ends up in model input.
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{i}", (w - 70, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frames
