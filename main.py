"""
Football Match Analysis Pipeline — Main Entry Point

Orchestrates the full pipeline:
1. Video loading & FPS detection
2. YOLO object detection + ByteTrack tracking (with video-specific stub caching)
3. Camera movement estimation & position adjustment
4. Player trajectory stabilization & persistent ID locking across overlaps
5. Ball interpolation
6. Ball possession assignment
7. Annotated video output with foot anchor ellipses and possession stats HUD
"""

import argparse
import os
import cv2
import numpy as np

from utils import read_video, save_video, get_video_fps
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from annotations import Annotator


def assert_contract(tracks, num_video_frames):
    """
    Verify CONTRACT.md structural invariants.
    Raises AssertionError with a descriptive message on any violation.
    """
    for key in ("players", "referees", "ball"):
        assert key in tracks, f"tracks missing required key '{key}'"
        assert len(tracks[key]) == num_video_frames, (
            f"tracks['{key}'] has {len(tracks[key])} entries "
            f"but video has {num_video_frames} frames"
        )

    # Ball key must be integer 1 (not string "1")
    for frame_num, ball_dict in enumerate(tracks["ball"]):
        for tid in ball_dict:
            assert tid == 1 and isinstance(tid, int), (
                f"frame {frame_num}: ball track_id must be int 1, got {tid!r}"
            )


def main():
    parser = argparse.ArgumentParser(description="Football Match Analysis Pipeline")
    parser.add_argument(
        "--input",
        default="input_videos/08fd33_4.mp4",
        help="Path to input video file (default: input_videos/08fd33_4.mp4)",
    )
    parser.add_argument(
        "--output",
        default="output_videos/output_video.avi",
        help="Path to output video file (default: output_videos/output_video.avi)",
    )
    args = parser.parse_args()

    video_path = args.input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path!r}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    tracks_stub_path = f"stubs/{video_name}_tracks_stub.pkl"
    camera_stub_path = f"stubs/{video_name}_camera_movement_stub.pkl"

    # ── 1. Read Video ───────────────────────────────────────────────────
    fps = get_video_fps(video_path)
    print(f"Loading video: {video_path}...")
    video_frames = read_video(video_path)
    num_frames = len(video_frames)
    duration_s = num_frames / fps if fps > 0 else 0
    print(
        f"Loaded {num_frames} frames ({video_frames[0].shape[1]}x{video_frames[0].shape[0]} @ {fps:.2f} fps, {duration_s:.1f}s)"
    )

    # ── 2. Initialize Tracker & Get Object Tracks ───────────────────────
    tracker = Tracker("models/best.pt")

    print(f"Retrieving object tracks (stub: {tracks_stub_path})...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=tracks_stub_path,
    )

    # Validate contract invariants from CONTRACT.md
    assert_contract(tracks, num_frames)

    # Add foot/center positions to all tracked objects
    tracker.add_position_to_tracks(tracks)

    # ── 3. Camera Movement Estimation ───────────────────────────────────
    print(f"Estimating camera movement (stub: {camera_stub_path})...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=camera_stub_path,
    )

    # Adjust positions by subtracting camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # ── 4. Stabilize Player IDs Across Occlusions & Overlaps ────────────
    print("Stabilizing player trajectories and locking IDs across overlaps...")
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

    # Run motion- and appearance-constrained Hungarian tracking to lock player IDs
    tracker.stabilize_player_ids(
        video_frames,
        tracks,
        camera_movement=camera_movement_per_frame,
        team_assigner=team_assigner,
    )

    # Re-calculate anchor positions for stabilized player tracks
    tracker.add_position_to_tracks(tracks)
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # ── 5. Interpolate Ball Positions ───────────────────────────────────
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ── 6. Ball Possession Assignment ───────────────────────────────────
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_dict = tracks["ball"][frame_num]
        ball_bbox = ball_dict[1]["bbox"] if 1 in ball_dict else [0, 0, 0, 0]
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

    # ── 7. Draw Visual Annotations (Ellipses & Possession HUD) ──────────
    print("Rendering visual annotations with Annotator (ellipses, triangles, badges)...")
    annotator = Annotator()
    output_video_frames = annotator.draw_annotations(video_frames, tracks)

    # Draw possession statistics HUD overlay on each frame
    for frame_num, frame in enumerate(output_video_frames):
        tracker.draw_team_ball_control(frame, frame_num, team_ball_control)

    output_path = args.output
    save_video(output_video_frames, output_path, fps=fps)

    print(f"Pipeline complete. Output saved to {output_path}")


if __name__ == "__main__":
    main()
