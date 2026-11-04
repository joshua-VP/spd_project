"""
Run Person A (Object Detection, Entity Tracking & Camera Motion) on Real Footage

Executes:
1. Load real FIFA match video (input_videos/input_video.mp4)
2. Fine-tuned YOLOv8x inference (models/best.pt) on 4 classes:
     0: ball, 1: goalkeeper, 2: player, 3: referee
3. ByteTrack persistent track ID assignment with goalkeeper->player override
4. Ball position interpolation
5. Optical flow camera movement estimation on static stadium banners
6. Coordinate compensation (subtracting camera pan from player foot coordinates)
7. Visual rendering with tracking ellipses, ball marker, and camera movement HUD

Output: output_videos/real_match_person_a.avi
"""

import os
import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator


def draw_person_a_visuals(tracker, frames, tracks, camera_movement):
    """
    Render Person A visual annotations:
    - Ground ellipses for players and referees with persistent ByteTrack IDs
    - Green triangle tracking the football
    - Telemetry HUD showing real-time camera panning vectors (dx, dy)
    """
    output_frames = []

    for frame_idx, frame in enumerate(frames):
        canvas = frame.copy()

        # 1. Draw Player Foot Ellipses & IDs
        player_dict = tracks["players"][frame_idx]
        for track_id, p_info in player_dict.items():
            bbox = p_info["bbox"]
            color = (255, 200, 0)
            canvas = tracker.draw_ellipse(canvas, bbox, color, track_id)

        # 2. Draw Referees
        ref_dict = tracks["referees"][frame_idx]
        for track_id, r_info in ref_dict.items():
            bbox = r_info["bbox"]
            color = (0, 255, 255)
            canvas = tracker.draw_ellipse(canvas, bbox, color)

        # 3. Draw Ball Tracker Triangle
        ball_dict = tracks["ball"][frame_idx]
        if 1 in ball_dict:
            b_box = ball_dict[1]["bbox"]
            bx = int((b_box[0] + b_box[2]) / 2)
            by = int(b_box[1])
            pts = np.array([[bx, by], [bx - 8, by - 16], [bx + 8, by - 16]])
            cv2.drawContours(canvas, [pts], 0, (0, 255, 0), cv2.FILLED)
            cv2.drawContours(canvas, [pts], 0, (0, 0, 0), 1)

        # 4. Draw Camera Movement Telemetry HUD
        dx, dy = camera_movement[frame_idx]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (20, 20), (520, 115), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        cv2.putText(
            canvas,
            "PERSON A: REAL VIDEO TRACKING & CAMERA ESTIMATION",
            (35, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Frame: {frame_idx:03d} | Active Tracks: {len(player_dict)} players",
            (35, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        direction = (
            "Camera Panning Right ->"
            if dx < -2
            else ("<- Camera Panning Left" if dx > 2 else "Camera Stable")
        )
        cv2.putText(
            canvas,
            f"Optical Flow: dx={dx:+.1f}px, dy={dy:+.1f}px ({direction})",
            (35, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 255, 100),
            1,
        )

        output_frames.append(canvas)

    return output_frames


def main():
    print("=" * 70)
    print("  PERSON A: Executing on Real Broadcast Match Footage")
    print("=" * 70)

    video_path = "input_videos/input_video.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing input video at {video_path}")

    # 1. Read Video
    print(f"\n1. Loading video: {video_path}...")
    frames = read_video(video_path)
    print(f"   -> Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})")

    # 2. YOLO Detection & ByteTrack Persistent Tracking
    print("\n2. Initializing YOLOv8 tracker with 'models/best.pt'...")
    tracker = Tracker("models/best.pt")

    print("   Running YOLO batch inference & ByteTrack tracking...")
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=True,
        stub_path="stubs/real_tracks_stub.pkl",
    )

    total_detections = sum(len(f) for f in tracks["players"])
    print(f"   -> Tracked {total_detections} player instances across {len(frames)} frames")

    # Add ground/foot anchor points
    tracker.add_position_to_tracks(tracks)

    # 3. Camera Movement Estimation via Optical Flow on Stadium Banners
    print("\n3. Estimating broadcast camera movement via Lucas-Kanade optical flow...")
    camera_estimator = CameraMovementEstimator(frames[0])
    camera_movement = camera_estimator.get_camera_movement(
        frames,
        read_from_stub=True,
        stub_path="stubs/real_camera_movement_stub.pkl",
    )

    significant_moves = sum(1 for m in camera_movement if abs(m[0]) > 0 or abs(m[1]) > 0)
    print(f"   -> Camera motion detected in {significant_moves}/{len(frames)} frames")

    # 4. Compensate entity coordinates for camera panning
    print("\n4. Neutralizing camera panning from player ground coordinates...")
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    # 5. Interpolate ball trajectory
    print("5. Interpolating ball trajectory across occluded frames...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 6. Render visual video
    print("\n6. Rendering tracking ellipses, ball markers, and HUD overlay...")
    annotated_frames = draw_person_a_visuals(tracker, frames, tracks, camera_movement)

    output_path = "output_videos/real_match_person_a.avi"
    os.makedirs("output_videos", exist_ok=True)
    save_video(annotated_frames, output_path)

    print(f"\n[COMPLETE] Real video analysis successfully rendered!")
    print(f"  -> Output file: {os.path.abspath(output_path)}")
    print(f"  -> File size:   {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()

