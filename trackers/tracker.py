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
        self.model_path = model_path
        self.model = YOLO(model_path) if model_path and os.path.exists(model_path) else None
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.30,
            lost_track_buffer=60,
            minimum_matching_threshold=0.85,
            frame_rate=25,
            minimum_consecutive_frames=1,
        )

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
        ball_positions = [
            x.get(1, {}).get("bbox", [np.nan, np.nan, np.nan, np.nan])
            if len(x.get(1, {}).get("bbox", [])) == 4
            else [np.nan, np.nan, np.nan, np.nan]
            for x in ball_positions
        ]
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
        if self.model is None:
            raise FileNotFoundError(
                f"YOLO model weights file not found at '{self.model_path}'. "
                "Please train the model or ensure read_from_stub=True with a valid stub file."
            )

        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i : i + batch_size], conf=0.25, iou=0.35
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

            # Class-agnostic NMS to eliminate duplicate overlapping bounding boxes on the same player
            detection_supervision = detection_supervision.with_nms(
                threshold=0.35, class_agnostic=True
            )

            # Override goalkeeper class ID (1) → player class ID (2)
            # so goalkeepers are tracked seamlessly with outfield players
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names.get(class_id) == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv[
                        "player"
                    ]

            # Filter: Pass ONLY players and referees to ByteTrack with conf >= 0.30
            # Exclude ball so fast ball trajectory does not fragment or steal player track IDs
            is_player = detection_supervision.class_id == cls_names_inv.get("player")
            is_referee = detection_supervision.class_id == cls_names_inv.get("referee")
            is_person = is_player | is_referee

            # Confidence threshold: players & referees must have conf >= 0.30
            valid_person_mask = is_person & (detection_supervision.confidence >= 0.30)
            det_persons = detection_supervision[valid_person_mask]

            # Run ByteTrack to assign persistent IDs
            detection_with_tracks = self.tracker.update_with_detections(det_persons)

            # Build per-frame track dictionaries
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if track_id is None:
                    continue
                track_id = int(track_id)

                if cls_id == cls_names_inv.get("player"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv.get("referee"):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Handle ball separately (no track ID, use fixed ID=1)
            # Take highest-confidence ball detection above 0.15 threshold
            ball_mask = (detection_supervision.class_id == cls_names_inv.get("ball")) & (
                detection_supervision.confidence >= 0.15
            )
            ball_detections = detection_supervision[ball_mask]
            if len(ball_detections) > 0:
                best_ball_idx = int(np.argmax(ball_detections.confidence))
                best_ball_bbox = ball_detections.xyxy[best_ball_idx].tolist()
                tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox}

        # Apply Exponential Moving Average (EMA) smoothing to eliminate inter-frame jitter
        self.smooth_player_tracks(tracks)

        # Save tracks to stub for future fast loading
        if stub_path is not None:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir:
                os.makedirs(stub_dir, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def smooth_player_tracks(self, tracks, alpha=0.65):
        """
        Apply Exponential Moving Average (EMA) to player bounding boxes
        to eliminate inter-frame detection jitter and coordinate shaking.
        """
        prev_positions = {}
        for frame_num, player_dict in enumerate(tracks["players"]):
            for track_id, p_info in player_dict.items():
                curr_box = np.array(p_info["bbox"], dtype=float)
                if track_id in prev_positions:
                    smoothed_box = alpha * curr_box + (1.0 - alpha) * prev_positions[track_id]
                else:
                    smoothed_box = curr_box
                prev_positions[track_id] = smoothed_box
                player_dict[track_id]["bbox"] = smoothed_box.tolist()

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

        # Draw foot anchor ellipse (semi-axes: half width and 0.20 width)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(max(10, int(width * 0.50)), max(4, int(width * 0.18))),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Draw track ID label rectangle + text
        if track_id is not None:
            rectangle_width = 34
            rectangle_height = 18
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = y2 + 4
            y2_rect = y2 + 4 + rectangle_height

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x_center - 8), int(y1_rect + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
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
        Positions adaptively to any frame resolution (720p, 1080p, etc.).

        Args:
            frame (np.ndarray): Video frame to draw on.
            frame_num (int): Current frame index.
            team_ball_control (np.ndarray): Array of team IDs (1 or 2) per frame.

        Returns:
            np.ndarray: Frame with possession overlay drawn.
        """
        h, w = frame.shape[:2]
        box_w = int(w * 0.35)
        box_h = int(h * 0.12)
        x2 = w - 20
        x1 = x2 - box_w
        y2 = h - 20
        y1 = y2 - box_h

        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
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

        font_scale = 0.55 if w <= 1280 else 0.85
        thickness = 2
        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (x1 + 15, y1 + int(box_h * 0.42)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (x1 + 15, y1 + int(box_h * 0.82)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
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

