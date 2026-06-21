from utils import *
from trackers import *
from court_line_detector import *
import cv2

def main():
    print("Main method")
    #video input
    input_video_path = "input_videos/input_video.avi" #mora .avi , nikako mp4
    video_frames = read_video(input_video_path)

#enumerate frames
    for i, frame in enumerate(video_frames):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{i}", (w - 50, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #player & ball detection
    player_tracker = PlayerTracker(model_path= "yolov8x.pt")
    ball_tracker = BallTracker(model_path= "models/yolo5_last.pt")
    court_line_detector = CourtLineDetector(model_path= "models/keypoints_model.pth")
    court_line_detections = court_line_detector.predict(video_frames[0])
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    player_detections = player_tracker.choose_and_filter_players(court_line_detections, player_detections)
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    #drawing detections
    output_frames = player_tracker.draw_boxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_boxes(output_frames, ball_detections)
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_line_detections)



    save(output_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()