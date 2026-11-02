from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    """
    YOLO-based object tracker for football match analysis.

    Uses a fine-tuned YOLO model for detection and ByteTrack for
    persistent object tracking across frames.

    Classes:
        0: ball
        1: goalkeeper
        2: player
        3: referee
    """

    def __init__(self, model_path):
        """
        Initialize tracker with YOLO model and ByteTrack.

        Args:
            model_path (str): Path to YOLO weights file (e.g., 'models/best.pt').
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        """
        Add foot position coordinates to each tracked object.

        For the ball, uses center of bounding box. For players, goalkeepers,
        and referees, uses the foot position (bottom-center of bbox).

        Args:
            tracks (dict): The tracks dictionary with bounding box data.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object_type == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions using pandas linear interpolation.

        The ball is often missed by detection (occlusion, blur, etc.).
        This fills gaps with linear interpolation and backfills any remaining
        NaN values at the start of the sequence.

        Args:
            ball_positions (list[dict]): List of per-frame ball track dicts.
                Each entry is {track_id: {"bbox": [x1,y1,x2,y2]}} or {}.

        Returns:
            list[dict]: Ball positions with interpolated bounding boxes.
        """
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames):
        """
        Run YOLO batch predictions on video frames.

        Processes frames in batches of 20 for efficient GPU utilization.

        Args:
            frames (list[np.ndarray]): List of video frame arrays.

        Returns:
            list: List of YOLO detection results, one per frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i : i + batch_size], conf=0.1
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get persistent object tracks across all video frames.

        Can load precomputed tracks from a pickle stub for fast iteration,
        or run full detection + tracking pipeline.

        Goalkeeper detections (class 1) are overridden to player class (2)
        so they are tracked seamlessly alongside outfield players.

        Args:
            frames (list[np.ndarray]): List of video frame arrays.
            read_from_stub (bool): If True, attempt to load tracks from stub_path.
            stub_path (str): Path to pickle file for saving/loading tracks.

        Returns:
            dict: Tracks dictionary with structure:
                {
                    "players": [  # list indexed by frame number
                        {track_id: {"bbox": [x1,y1,x2,y2]}, ...},
                        ...
                    ],
                    "referees": [...],
                    "ball": [...]
                }
        """
        # Check for pre-computed stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        # Run YOLO detection on all frames
        detections = self.detect_frames(frames)

        # Initialize tracks structure
        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # e.g., {0: 'ball', 1: 'goalkeeper', ...}
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert YOLO results to supervision Detections
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Override goalkeeper class ID (1) → player class ID (2)
            # so goalkeepers are tracked seamlessly with outfield players
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv[
                        "player"
                    ]

            # Run ByteTrack to assign persistent IDs
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            # Build per-frame track dictionaries
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Handle ball separately (no track ID, use fixed ID=1)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save tracks to stub for future fast loading
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse annotation at the foot of a detected player/referee.

        Args:
            frame (np.ndarray): Video frame to draw on.
            bbox (list): Bounding box [x1, y1, x2, y2].
            color (tuple): BGR color tuple.
            track_id (int, optional): Track ID to display in a label.

        Returns:
            np.ndarray: Frame with ellipse drawn.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        # Draw track ID label rectangle + text
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw a triangle marker above the ball position.

        Args:
            frame (np.ndarray): Video frame to draw on.
            bbox (list): Bounding box [x1, y1, x2, y2].
            color (tuple): BGR color tuple.

        Returns:
            np.ndarray: Frame with triangle drawn.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [[x, y], [x - 10, y - 20], [x + 10, y - 20]]
        )

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw a semi-transparent ball possession statistics overlay.

        Args:
            frame (np.ndarray): Video frame to draw on.
            frame_num (int): Current frame index.
            team_ball_control (np.ndarray): Array of team IDs (1 or 2) per frame.

        Returns:
            np.ndarray: Frame with possession overlay drawn.
        """
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate possession percentages up to current frame
        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        team_1_num_frames = (team_ball_control_till_frame == 1).sum()
        team_2_num_frames = (team_ball_control_till_frame == 2).sum()
        total = team_1_num_frames + team_2_num_frames
        if total == 0:
            team_1 = 0
            team_2 = 0
        else:
            team_1 = team_1_num_frames / total
            team_2 = team_2_num_frames / total

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draw all visual annotations (ellipses, triangles, possession stats)
        onto each video frame.

        Args:
            video_frames (list[np.ndarray]): Raw video frames.
            tracks (dict): Object tracks dictionary.
            team_ball_control (np.ndarray): Per-frame team possession array.

        Returns:
            list[np.ndarray]: Annotated video frames.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw ball possession stats
            frame = self.draw_team_ball_control(
                frame, frame_num, team_ball_control
            )

            output_video_frames.append(frame)

        return output_video_frames

