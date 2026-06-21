import cv2

def save(video_frames, output_path, fps=24):
    if not video_frames:
        return
    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in video_frames:
        out.write(frame)
    out.release()

#returns video as array of frames
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames