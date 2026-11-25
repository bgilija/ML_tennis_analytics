import argparse
import os

from utils import read_video, save, draw_frame_numbers, compute_court_homography, reprojection_errors_px
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from analysis import (
    detect_hits,
    detect_bounces,
    build_ball_ground_track,
    player_court_positions,
    compute_shot_speeds,
    compute_frame_stats,
    draw_stats,
)


def parse_args():
    p = argparse.ArgumentParser(description="Tennis video analysis")
    p.add_argument("--input", default="input_videos/input_video.avi", help="input video (.avi)")
    p.add_argument("--output", default=None, help="output video (default: output_videos/<name>_output.avi)")
    p.add_argument("--use-stubs", action="store_true", help="reuse cached detections from tracker_stubs/")
    p.add_argument("--player-model", default="yolov8x.pt")
    p.add_argument("--ball-model", default="models/yolo5_last.pt")
    p.add_argument("--court-model", default="models/keypoints_model.pth")
    return p.parse_args()


def run(args):
    name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or f"output_videos/{name}_output.avi"

    video_frames, fps = read_video(args.input)

    # --- detection (frames must still be clean here)
    court_line_detector = CourtLineDetector(model_path=args.court_model)
    court_keypoints = court_line_detector.predict_video(video_frames)
    H, inliers = compute_court_homography(court_keypoints)
    errors = reprojection_errors_px(court_keypoints, H)
    print(f"Court keypoints: {inliers.sum()}/14 inliers, mean reprojection error {errors.mean():.1f}px")

    player_tracker = PlayerTracker(model_path=args.player_model)
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=args.use_stubs, stub_path=f"tracker_stubs/{name}_players.pkl")
    player_detections = player_tracker.choose_and_filter_players(H, player_detections)

    ball_tracker = BallTracker(model_path=args.ball_model)
    ball_candidates = ball_tracker.detect_frames(
        video_frames, read_from_stub=args.use_stubs, stub_path=f"tracker_stubs/{name}_ball.pkl")
    ball_detections = ball_tracker.select_ball_candidates(ball_candidates)
    ball_detections = ball_tracker.remove_outliers(ball_detections)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # --- analysis
    hits = detect_hits(ball_detections, player_detections)
    bounces = detect_bounces(ball_detections, hits, H)
    ball_ground = build_ball_ground_track(len(video_frames), hits, bounces, H)
    player_positions = player_court_positions(player_detections, H)
    shots = compute_shot_speeds(hits, bounces, H, fps)
    frame_stats = compute_frame_stats(len(video_frames), player_positions, shots, fps)

    print(f"Hits: {[(h['frame'], h['player_id']) for h in hits]}")
    print(f"Bounces: {[(b['frame'], 'in' if b['in'] else 'out') for b in bounces]}")
    print(f"Shot speeds (km/h): {[round(s['speed_kmh']) for s in shots]}")

    # --- drawing
    mini_court = MiniCourt(video_frames[0])
    output_frames = player_tracker.draw_boxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_boxes(output_frames, ball_detections)
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_keypoints)
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_bounces(output_frames, bounces)
    output_frames = mini_court.draw_player_ball_positions(output_frames, player_positions, ball_ground)
    output_frames = draw_stats(output_frames, frame_stats)
    output_frames = draw_frame_numbers(output_frames)

    save(output_frames, output_path, fps)
    print(f"Saved {output_path}")
    return {'hits': hits, 'bounces': bounces, 'shots': shots, 'players': player_detections,
            'ball': ball_detections, 'H': H, 'keypoint_errors': errors}


if __name__ == "__main__":
    run(parse_args())
