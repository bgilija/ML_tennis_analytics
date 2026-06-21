from ultralytics import YOLO
import cv2
import pickle
import os
import sys

sys.path.append('../')
from utils import measure_distance, get_center_of_bounding_box

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players=self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections=[]
        for player_dictionary in player_detections:
            filtered_player_dict={track_id: bounding_box for track_id,bounding_box in player_dictionary.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections
    def choose_players(self, court_keypoints, player_detections): #myb needs a fix to choose by centers of court sides
        distances = []
        for track_id, bounding_box in player_detections.items():
            player_center = get_center_of_bounding_box(bounding_box)
            min_dist = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                dist = measure_distance(player_center, court_keypoint)
                if dist < min_dist:
                    min_dist = dist
            distances.append((track_id, min_dist))

        distances.sort(key=lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frame(self, frame):
        results = self.model.track(frame, persist= True)[0]
        id_name_dict = results.names

        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict



    def detect_frames(self, frames, read_from_stub= False, stub_path = None):
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def draw_boxes(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bounding_box in player_dict.items():
                x1, y1, x2, y2 = map(int, bounding_box)
                cv2.putText(frame, f"Player Id: {track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            output_frames.append(frame)
        return output_frames