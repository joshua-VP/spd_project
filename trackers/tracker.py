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

    def __init__(self, model_path=None):
        """
        Initialize tracker with YOLO model and ByteTrack.

        Args:
            model_path (str, optional): Path to YOLO weights file (e.g., 'models/best.pt').
        """
        self.model_path = model_path
        self.model = YOLO(model_path) if model_path and os.path.exists(model_path) else None
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            minimum_matching_threshold=0.60,
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
        total_batches = (len(frames) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(frames), batch_size)):
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0 or (batch_idx + 1) == total_batches:
                print(f"   YOLO batch {batch_idx + 1}/{total_batches} (frames {i}-{min(len(frames), i + batch_size)})...", flush=True)
            detections_batch = self.model.predict(
                frames[i : i + batch_size], conf=0.25, iou=0.35, verbose=False
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

    def stitch_and_stabilize_tracks(
        self,
        tracks,
        camera_movement=None,
        max_gap=45,
        max_distance=80,
        min_track_length=2,
    ):
        """
        Stitch fragmented tracklets across occlusions without dropping overlapping players.
        Uses camera movement compensation so panning does not cause tracklet mismatch,
        and enforces a temporal overlap guard so two players in the same frame never merge.
        """
        import copy
        players = tracks["players"]
        num_frames = len(players)
        if num_frames == 0:
            return tracks

        # Compute cumulative camera displacement for pitch-space reference
        if camera_movement is not None and len(camera_movement) == num_frames:
            cum_cam = np.cumsum(camera_movement, axis=0)
        else:
            cum_cam = np.zeros((num_frames, 2))

        # 1. Group frame observations into tracklet records
        tracklets = {}
        for f_idx, frame_players in enumerate(players):
            cam_offset = cum_cam[f_idx]
            for tid, info in frame_players.items():
                box = info["bbox"]
                # Compensated pitch center
                global_center = (
                    (box[0] + box[2]) / 2.0 - cam_offset[0],
                    (box[1] + box[3]) / 2.0 - cam_offset[1],
                )
                if tid not in tracklets:
                    tracklets[tid] = {
                        "id": tid,
                        "start_f": f_idx,
                        "end_f": f_idx,
                        "start_pos": global_center,
                        "end_pos": global_center,
                        "frames": {f_idx: copy.copy(info)},
                    }
                else:
                    tracklets[tid]["end_f"] = f_idx
                    tracklets[tid]["end_pos"] = global_center
                    tracklets[tid]["frames"][f_idx] = copy.copy(info)

        # 2. Filter out ephemeral noise (< min_track_length)
        valid_tracklets = {
            tid: t
            for tid, t in tracklets.items()
            if len(t["frames"]) >= min_track_length
        }

        # 3. Stitch tracklets using Union-Find with temporal collision protection
        sorted_tids = sorted(
            valid_tracklets.keys(), key=lambda x: valid_tracklets[x]["start_f"]
        )
        id_map = {tid: tid for tid in sorted_tids}
        root_frames = {tid: set(valid_tracklets[tid]["frames"].keys()) for tid in sorted_tids}

        def get_root(tid):
            while id_map[tid] != tid:
                id_map[tid] = id_map[id_map[tid]]
                tid = id_map[tid]
            return tid

        for i, tid1 in enumerate(sorted_tids):
            t1 = valid_tracklets[tid1]
            root1 = get_root(tid1)
            best_candidate = None
            min_dist = float("inf")

            for j in range(i + 1, len(sorted_tids)):
                tid2 = sorted_tids[j]
                t2 = valid_tracklets[tid2]
                root2 = get_root(tid2)
                if root1 == root2:
                    continue

                # NEVER merge two track clusters if they share ANY active video frames
                if root_frames[root1].intersection(root_frames[root2]):
                    continue

                gap = t2["start_f"] - t1["end_f"]
                if 1 <= gap <= max_gap:
                    d = np.hypot(
                        t2["start_pos"][0] - t1["end_pos"][0],
                        t2["start_pos"][1] - t1["end_pos"][1],
                    )
                    if d <= max_distance and d < min_dist:
                        min_dist = d
                        best_candidate = root2
                elif gap > max_gap:
                    break

            if best_candidate is not None:
                id_map[best_candidate] = root1
                root_frames[root1].update(root_frames[best_candidate])

        # 4. Clean re-numbering for consistent player IDs
        unique_roots = []
        for tid in sorted_tids:
            r = get_root(tid)
            if r not in unique_roots:
                unique_roots.append(r)
        root_to_clean_id = {r: i + 1 for i, r in enumerate(unique_roots)}

        # 5. Rebuild tracks with unified canonical IDs
        stitched_players = [{} for _ in range(num_frames)]
        for tid, t in valid_tracklets.items():
            clean_id = root_to_clean_id[get_root(tid)]
            for f_idx, info in t["frames"].items():
                stitched_players[f_idx][clean_id] = copy.copy(info)

        tracks["players"] = stitched_players
        return tracks

    def interpolate_player_tracks(self, tracks, max_gap=3):
        """
        Linearly interpolate player bounding boxes across short occlusion gaps (1-3 frames).
        Prevents ellipses and ID labels from flickering or disappearing when players pass each other.
        """
        import copy
        from collections import defaultdict
        players = tracks["players"]
        num_frames = len(players)

        player_frames = defaultdict(dict)
        for f_idx, f_dict in enumerate(players):
            for tid, info in f_dict.items():
                player_frames[tid][f_idx] = info

        for tid, f_map in player_frames.items():
            sorted_f = sorted(f_map.keys())
            for idx in range(len(sorted_f) - 1):
                f1 = sorted_f[idx]
                f2 = sorted_f[idx + 1]
                gap = f2 - f1
                if 1 < gap <= (max_gap + 1):
                    b1 = np.array(f_map[f1]["bbox"], dtype=float)
                    b2 = np.array(f_map[f2]["bbox"], dtype=float)
                    template_info = copy.copy(f_map[f1])
                    for step in range(1, gap):
                        frac = step / gap
                        interp_box = (b1 + (b2 - b1) * frac).tolist()
                        fill_frame = f1 + step
                        new_info = copy.copy(template_info)
                        new_info["bbox"] = interp_box
                        players[fill_frame][tid] = new_info
        return tracks

    def stabilize_player_ids(
        self,
        video_frames,
        tracks,
        camera_movement=None,
        team_assigner=None,
        max_match_dist=85.0,
        max_lost_frames=25,
    ):
        """
        Optimal bipartite matching with velocity projection and jersey color preservation.
        Ensures each player's ID stays permanently attached even during crosses, tackles,
        and close overlaps.
        """
        import copy
        from scipy.optimize import linear_sum_assignment
        from team_assigner import TeamAssigner

        players = tracks["players"]
        num_frames = len(players)
        if num_frames == 0 or len(video_frames) == 0:
            return tracks

        # 1. Setup Team Assigner if not provided
        if team_assigner is None:
            team_assigner = TeamAssigner()
            ref_idx = 0
            for idx, p_track in enumerate(players):
                if len(p_track) >= 6:
                    ref_idx = idx
                    break
            team_assigner.assign_team_color(video_frames[ref_idx], players[ref_idx])

        c1 = team_assigner.team_colors.get(1, np.array([40, 40, 220]))
        c2 = team_assigner.team_colors.get(2, np.array([220, 220, 220]))

        # Cumulative camera displacement for pitch-space continuous tracking
        if camera_movement is not None and len(camera_movement) == num_frames:
            cum_cam = np.cumsum(camera_movement, axis=0)
        else:
            cum_cam = np.zeros((num_frames, 2))

        class PlayerTracklet:
            def __init__(self, tid, f_idx, box, pitch_pos, team_id, team_color):
                self.id = tid
                self.team_votes = {1: 0, 2: 0}
                if team_id in self.team_votes:
                    self.team_votes[team_id] += 1
                self.team = team_id
                self.team_color = team_color
                self.last_f = f_idx
                self.pos = np.array(pitch_pos, dtype=float)
                self.vel = np.zeros(2, dtype=float)
                self.box = list(box)
                self.frames_lost = 0

            def predict(self):
                return self.pos + self.vel * 0.95

            def update(self, f_idx, box, pitch_pos, team_id, team_color):
                new_pos = np.array(pitch_pos, dtype=float)
                dt = max(1, f_idx - self.last_f)
                inst_vel = (new_pos - self.pos) / dt
                self.vel = 0.70 * self.vel + 0.30 * inst_vel
                self.pos = new_pos
                self.box = list(box)
                self.last_f = f_idx
                self.frames_lost = 0
                if team_id in self.team_votes:
                    self.team_votes[team_id] += 1
                    self.team = max(self.team_votes, key=self.team_votes.get)
                    if team_assigner is not None and hasattr(team_assigner, "team_colors") and self.team in team_assigner.team_colors:
                        self.team_color = team_assigner.team_colors[self.team]

            def extrapolate(self, f_idx, cam_off):
                self.pos = self.pos + self.vel * 0.90
                w = self.box[2] - self.box[0]
                h = self.box[3] - self.box[1]
                foot_x = self.pos[0] + cam_off[0]
                foot_y = self.pos[1] + cam_off[1]
                self.box = [foot_x - w / 2.0, foot_y - h, foot_x + w / 2.0, foot_y]
                self.frames_lost += 1

        active_tracks = []
        next_id = 1
        stage1_players = [{} for _ in range(num_frames)]

        for f_idx in range(num_frames):
            cam_off = cum_cam[f_idx]
            frame = video_frames[f_idx]
            f_dets = players[f_idx]

            dets = []
            det_boxes = []
            det_teams = []
            det_colors = []

            for raw_id, info in f_dets.items():
                box = info["bbox"]
                foot_x = (box[0] + box[2]) / 2.0 - cam_off[0]
                foot_y = box[3] - cam_off[1]
                dets.append([foot_x, foot_y])
                det_boxes.append(box)

                # Fast torso median color
                b = [int(v) for v in box]
                w = max(1, b[2] - b[0])
                h = max(1, b[3] - b[1])
                y_start = max(0, b[1] + int(h * 0.15))
                y_end = min(frame.shape[0], b[1] + int(h * 0.45))
                x_start = max(0, b[0] + int(w * 0.25))
                x_end = min(frame.shape[1], b[0] + int(w * 0.75))
                torso = frame[y_start:y_end, x_start:x_end]
                if torso.size > 0:
                    med_color = np.median(torso, axis=(0, 1))
                    d1 = np.linalg.norm(med_color - c1)
                    d2 = np.linalg.norm(med_color - c2)
                    team = 1 if d1 < d2 else 2
                else:
                    team = 1
                det_teams.append(team)
                det_colors.append(team_assigner.team_colors[team])

            active_tracks = [t for t in active_tracks if t.frames_lost <= max_lost_frames]

            if len(dets) == 0:
                for t in active_tracks:
                    t.extrapolate(f_idx, cam_off)
                    if t.frames_lost <= 3:
                        stage1_players[f_idx][t.id] = {
                            "bbox": [float(v) for v in t.box],
                            "team": t.team,
                            "team_color": t.team_color,
                        }
                continue

            if len(active_tracks) == 0:
                for d_idx in range(len(dets)):
                    t = PlayerTracklet(next_id, f_idx, det_boxes[d_idx], dets[d_idx], det_teams[d_idx], det_colors[d_idx])
                    next_id += 1
                    active_tracks.append(t)
                    stage1_players[f_idx][t.id] = {
                        "bbox": [float(v) for v in t.box],
                        "team": t.team,
                        "team_color": t.team_color,
                    }
                continue

            # Hungarian Cost Matrix with velocity projection & team color preservation
            cost = np.full((len(active_tracks), len(dets)), 1e6, dtype=float)
            for t_idx, t in enumerate(active_tracks):
                pred = t.predict()
                for d_idx in range(len(dets)):
                    dist = np.hypot(pred[0] - dets[d_idx][0], pred[1] - dets[d_idx][1])
                    if dist < max_match_dist:
                        team_pen = 0.0 if t.team == det_teams[d_idx] else 150.0
                        cost[t_idx, d_idx] = dist + team_pen

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_t = set()
            matched_d = set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 200.0:
                    active_tracks[r].update(f_idx, det_boxes[c], dets[c], det_teams[c], det_colors[c])
                    stage1_players[f_idx][active_tracks[r].id] = {
                        "bbox": [float(v) for v in active_tracks[r].box],
                        "team": active_tracks[r].team,
                        "team_color": active_tracks[r].team_color,
                    }
                    matched_t.add(r)
                    matched_d.add(c)

            # Extrapolate unmatched active tracks for up to 3 frames during momentary overlap
            for t_idx, t in enumerate(active_tracks):
                if t_idx not in matched_t:
                    t.extrapolate(f_idx, cam_off)
                    if t.frames_lost <= 3:
                        stage1_players[f_idx][t.id] = {
                            "bbox": [float(v) for v in t.box],
                            "team": t.team,
                            "team_color": t.team_color,
                        }

            # Unmatched detections get new IDs
            for d_idx in range(len(dets)):
                if d_idx not in matched_d:
                    t = PlayerTracklet(next_id, f_idx, det_boxes[d_idx], dets[d_idx], det_teams[d_idx], det_colors[d_idx])
                    next_id += 1
                    active_tracks.append(t)
                    stage1_players[f_idx][t.id] = {
                        "bbox": [float(v) for v in t.box],
                        "team": t.team,
                        "team_color": t.team_color,
                    }

        # Stage 2: Tracklet stitching across longer gaps (team-aware, collision-free)
        tracklets = {}
        for f_idx, f_dict in enumerate(stage1_players):
            cam_off = cum_cam[f_idx]
            for tid, info in f_dict.items():
                box = info["bbox"]
                gc = ((box[0] + box[2]) / 2.0 - cam_off[0], box[3] - cam_off[1])
                if tid not in tracklets:
                    tracklets[tid] = {
                        "id": tid,
                        "team": info["team"],
                        "team_color": info["team_color"],
                        "start_f": f_idx,
                        "end_f": f_idx,
                        "start_pos": gc,
                        "end_pos": gc,
                        "frames": {f_idx: info},
                    }
                else:
                    tracklets[tid]["end_f"] = f_idx
                    tracklets[tid]["end_pos"] = gc
                    tracklets[tid]["frames"][f_idx] = info

        sorted_tids = sorted(tracklets.keys(), key=lambda x: tracklets[x]["start_f"])
        id_map = {tid: tid for tid in sorted_tids}
        root_frames = {tid: set(tracklets[tid]["frames"].keys()) for tid in sorted_tids}

        def get_root(tid):
            while id_map[tid] != tid:
                id_map[tid] = id_map[id_map[tid]]
                tid = id_map[tid]
            return tid

        for i, tid1 in enumerate(sorted_tids):
            t1 = tracklets[tid1]
            root1 = get_root(tid1)
            best_candidate = None
            min_dist = float("inf")

            for j in range(i + 1, len(sorted_tids)):
                tid2 = sorted_tids[j]
                t2 = tracklets[tid2]
                root2 = get_root(tid2)
                if root1 == root2:
                    continue
                if root_frames[root1].intersection(root_frames[root2]):
                    continue

                gap = t2["start_f"] - t1["end_f"]
                if 1 <= gap <= 90:
                    d = np.hypot(t2["start_pos"][0] - t1["end_pos"][0], t2["start_pos"][1] - t1["end_pos"][1])
                    max_d = min(220.0, 90.0 + gap * 12.0)
                    team_match = (t1["team"] == t2["team"])
                    if (team_match and d <= max_d) or (d <= 70.0):
                        cost = d + (0.0 if team_match else 50.0)
                        if cost < min_dist:
                            min_dist = cost
                            best_candidate = root2
                elif gap > 90:
                    break

            if best_candidate is not None:
                id_map[best_candidate] = root1
                root_frames[root1].update(root_frames[best_candidate])

        unique_roots = []
        for tid in sorted_tids:
            r = get_root(tid)
            if r not in unique_roots:
                unique_roots.append(r)
        root_to_clean_id = {r: i + 1 for i, r in enumerate(unique_roots)}

        final_players = [{} for _ in range(num_frames)]
        for tid, t in tracklets.items():
            clean_id = root_to_clean_id[get_root(tid)]
            for f_idx, info in t["frames"].items():
                info_copy = dict(info)
                final_players[f_idx][clean_id] = info_copy

        tracks["players"] = final_players

        # Smooth bounding boxes to eliminate inter-frame coordinate shaking
        self.smooth_player_tracks(tracks, alpha=0.70)

        return tracks


    def draw_bbox(self, frame, bbox, color, track_id=None, has_ball=False):
        """
        Draw a clean, professional player bounding box with track ID badge.

        Args:
            frame (np.ndarray): Video frame to draw on.
            bbox (list): Bounding box [x1, y1, x2, y2].
            color (tuple): BGR color tuple.
            track_id (int, optional): Track ID to display.
            has_ball (bool): Whether this player possesses the ball.

        Returns:
            np.ndarray: Frame with bounding box drawn.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        # Main box border: red highlight if player has ball
        box_color = (0, 0, 255) if has_ball else color
        thickness = 3 if has_ball else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)

        # Draw ID badge pill at top of bounding box
        if track_id is not None:
            text = f"#{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            text_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

            badge_h = text_h + 8
            badge_w = text_w + 10

            # Position badge above box if within frame, otherwise tuck inside top
            if y1 >= badge_h:
                b_y1 = y1 - badge_h
                b_y2 = y1
            else:
                b_y1 = y1
                b_y2 = y1 + badge_h

            b_x1 = x1
            b_x2 = min(frame_w - 1, x1 + badge_w)

            cv2.rectangle(frame, (b_x1, b_y1), (b_x2, b_y2), box_color, cv2.FILLED)

            # High-contrast text based on badge luminance
            lum = 0.114 * box_color[0] + 0.587 * box_color[1] + 0.299 * box_color[2]
            text_color = (0, 0, 0) if lum > 135 else (255, 255, 255)

            cv2.putText(
                frame,
                text,
                (b_x1 + 5, b_y2 - 5),
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )

        return frame

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

    def draw_annotations(
        self, video_frames, tracks, team_ball_control, draw_mode="bbox"
    ):
        """
        Draw all visual annotations (bounding boxes, triangles, possession stats)
        onto each video frame.

        Args:
            video_frames (list[np.ndarray]): Raw video frames.
            tracks (dict): Object tracks dictionary.
            team_ball_control (np.ndarray): Per-frame team possession array.
            draw_mode (str): 'bbox' (default, clean bounding boxes with top ID badges),
                             'ellipse' (foot circles), or 'both'.

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
                has_ball = player.get("has_ball", False)

                if draw_mode in ("bbox", "both"):
                    frame = self.draw_bbox(
                        frame, player["bbox"], color, track_id, has_ball=has_ball
                    )
                if draw_mode in ("ellipse", "both"):
                    frame = self.draw_ellipse(
                        frame, player["bbox"], color, track_id
                    )

                if has_ball and draw_mode == "ellipse":
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                ref_color = (0, 255, 255)
                if draw_mode in ("bbox", "both"):
                    frame = self.draw_bbox(
                        frame, referee["bbox"], ref_color, track_id=None
                    )
                if draw_mode in ("ellipse", "both"):
                    frame = self.draw_ellipse(frame, referee["bbox"], ref_color)

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw ball possession stats
            frame = self.draw_team_ball_control(
                frame, frame_num, team_ball_control
            )

            output_video_frames.append(frame)

        return output_video_frames

