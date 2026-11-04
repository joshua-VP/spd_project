"""
Football Match Analysis Pipeline — Main Entry Point

Orchestrates the full pipeline:
1. Video loading
2. YOLO object detection + ByteTrack tracking
3. Camera movement estimation & position adjustment
4. Ball interpolation
5. Team assignment via K-Means
6. Ball possession assignment
7. Annotated video output
"""

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import cv2
import numpy as np


def main():
    # ── 1. Read Video ───────────────────────────────────────────────────
    video_path = "input_videos/input_video.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    cap.release()

    video_frames = read_video(video_path)

    # ── 2. Initialize Tracker & Get Object Tracks ───────────────────────
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/tracks_stub.pkl",
    )

    # Add foot/center positions to all tracked objects
    tracker.add_position_to_tracks(tracks)

    # ── 3. Camera Movement Estimation ───────────────────────────────────
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/camera_movement_stub.pkl",
    )

    # Adjust positions by subtracting camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # Stabilize player tracklets: stitch broken tracks across occlusions & prune transient noise
    tracker.stitch_and_stabilize_tracks(tracks, camera_movement_per_frame)

    # ── 4. Interpolate Ball Positions ───────────────────────────────────
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ── 5. Team Assignment (K-Means on Jersey Colors) ───────────────────
    team_assigner = TeamAssigner()
    # Find a representative frame with multiple players to fit two team jersey clusters
    ref_frame_idx = 0
    for idx, p_track in enumerate(tracks["players"]):
        if len(p_track) >= 6:
            ref_frame_idx = idx
            break
        elif len(p_track) >= 2:
            ref_frame_idx = idx

    team_assigner.assign_team_color(
        video_frames[ref_frame_idx], tracks["players"][ref_frame_idx]
    )

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # ── 6. Ball Possession Assignment ───────────────────────────────────
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox
        )

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            # Carry forward previous possession
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 1
            )

    team_ball_control = np.array(team_ball_control)

    # ── 7. Draw Annotations & Save Output Video ────────────────────────
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control, draw_mode="bbox"
    )

    save_video(output_video_frames, "output_videos/output_video.avi", fps=fps)

    print("Pipeline complete. Output saved to output_videos/output_video.avi")


if __name__ == "__main__":
    main()

