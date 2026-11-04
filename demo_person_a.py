"""
Person A Visual Demonstration & Verification Script

Generates a synthetic football match sequence with camera panning motion,
runs CameraMovementEstimator to track camera movement, adjusts object positions,
renders all visual annotations (ellipses, triangles, camera HUD), and saves a
playable video to output_videos/person_a_demo.avi.

Run with:
    python demo_person_a.py
"""

import os
import cv2
import numpy as np
from utils import save_video
from camera_movement_estimator import CameraMovementEstimator
from generate_dummy_stub import generate_dummy_tracks


def create_synthetic_match_video(num_frames=60, width=1280, height=720):
    """
    Generate synthetic frames simulating a camera panning right across a football pitch.
    Static ad banners at top (0-20%) and bottom (80-100%) shift by 8px per frame.
    """
    frames = []
    print(f"Generating {num_frames} synthetic match frames ({width}x{height})...")

    # Base pitch texture
    pitch_green = (34, 139, 34)  # BGR Forest Green
    banner_top_color = (220, 50, 50)  # Red banner
    banner_bot_color = (50, 50, 220)  # Blue banner

    for frame_idx in range(num_frames):
        frame = np.full((height, width, 3), pitch_green, dtype=np.uint8)

        # Pan offset: simulates camera panning horizontally by 8 pixels per frame
        pan_offset = frame_idx * 8

        # Draw top advertising banner (0 to 20% height) with vertical texture stripes
        top_h = int(height * 0.20)
        cv2.rectangle(frame, (0, 0), (width, top_h), banner_top_color, -1)
        for x in range(-100, width + 100, 40):
            stripe_x = (x - pan_offset) % (width + 80)
            cv2.rectangle(frame, (stripe_x, 10), (stripe_x + 20, top_h - 10), (255, 255, 255), -1)
            cv2.putText(frame, "SPONSOR", (stripe_x - 10, top_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw bottom advertising banner (80% to 100% height) with alternating blocks
        bot_h_start = int(height * 0.80)
        cv2.rectangle(frame, (0, bot_h_start), (width, height), banner_bot_color, -1)
        for x in range(-100, width + 100, 60):
            block_x = (x - pan_offset) % (width + 120)
            cv2.rectangle(frame, (block_x, bot_h_start + 15), (block_x + 35, height - 15), (0, 255, 255), -1)

        # Pitch markings (grass stripes & center circle)
        for x in range(-200, width + 200, 160):
            line_x = x - pan_offset
            if 0 <= line_x <= width:
                cv2.line(frame, (line_x, top_h), (line_x, bot_h_start), (40, 160, 40), 25)

        # Center line & circle
        center_x = (width // 2) - pan_offset
        if -100 <= center_x <= width + 100:
            cv2.line(frame, (center_x, top_h), (center_x, bot_h_start), (255, 255, 255), 3)
            cv2.circle(frame, (center_x, height // 2), 70, (255, 255, 255), 3)

        frames.append(frame)

    return frames


def draw_hud(frame, frame_idx, camera_movement, ball_pos=None):
    """Draw telemetry overlay for verification."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (450, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    dx, dy = camera_movement[frame_idx]
    cv2.putText(frame, f"PERSON A VERIFICATION DEMO", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_idx:02d} | Optical Flow: dx={dx:+.1f}px, dy={dy:+.1f}px", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    status = "Panning Right" if dx < -2 else ("Panning Left" if dx > 2 else "Static")
    cv2.putText(frame, f"Detected Camera Motion: {status}", (35, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)


def main():
    print("=" * 65)
    print("  PERSON A: End-to-End Pipeline & Camera Estimator Verification")
    print("=" * 65)

    os.makedirs("output_videos", exist_ok=True)
    os.makedirs("stubs", exist_ok=True)

    # 1. Generate synthetic video with camera panning
    frames = create_synthetic_match_video(num_frames=60, width=1280, height=720)

    # 2. Estimate camera movement using Lucas-Kanade optical flow on stadium banner masks
    print("\nRunning CameraMovementEstimator on banner regions...")
    estimator = CameraMovementEstimator(frames[0])
    camera_movement = estimator.get_camera_movement(frames)

    # Calculate average detected panning
    movements_x = [m[0] for m in camera_movement[1:]]
    avg_pan_x = np.mean(movements_x)
    print(f"  [SUCCESS] Camera movement successfully estimated across {len(frames)} frames!")
    print(f"  Average estimated horizontal pan shift: {avg_pan_x:.2f} px/frame")

    # 3. Generate mock tracked entities
    print("\nSimulating tracked entities (players, referee, ball)...")
    tracks = generate_dummy_tracks(num_frames=60, num_players=6, num_referees=1)

    # 4. Add positions and adjust coordinates based on camera movement
    for obj_type, obj_list in tracks.items():
        for frame_dict in obj_list:
            for tid, tinfo in frame_dict.items():
                bb = tinfo["bbox"]
                if obj_type == "ball":
                    tinfo["position"] = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
                else:
                    tinfo["position"] = ((bb[0] + bb[2]) / 2, bb[3])

    estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    # 5. Render visual annotations (ellipses, triangles, possession)
    print("Rendering Person A visual annotations...")
    annotated_frames = []

    for frame_idx, frame in enumerate(frames):
        canvas = frame.copy()

        # Draw players with tracking ellipses
        for player_id, p_info in tracks["players"][frame_idx].items():
            bbox = p_info["bbox"]
            color = (255, 150, 0) if player_id % 2 == 0 else (0, 150, 255)
            x_center = int((bbox[0] + bbox[2]) / 2)
            y2 = int(bbox[3])
            width = int(bbox[2] - bbox[0])

            # Foot ellipse
            cv2.ellipse(canvas, (x_center, y2), (width // 2, int(0.35 * width)), 0, -45, 235, color, 2)
            # Track ID box
            cv2.rectangle(canvas, (x_center - 15, y2 + 5), (x_center + 15, y2 + 25), color, -1)
            cv2.putText(canvas, f"{player_id}", (x_center - 7, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        # Draw ball with triangle marker
        ball_info = tracks["ball"][frame_idx].get(1)
        if ball_info:
            bx = int((ball_info["bbox"][0] + ball_info["bbox"][2]) / 2)
            by = int(ball_info["bbox"][1])
            pts = np.array([[bx, by], [bx - 8, by - 16], [bx + 8, by - 16]])
            cv2.drawContours(canvas, [pts], 0, (0, 255, 0), cv2.FILLED)
            cv2.drawContours(canvas, [pts], 0, (0, 0, 0), 1)

        # Draw HUD
        draw_hud(canvas, frame_idx, camera_movement)
        annotated_frames.append(canvas)

    # 6. Save video
    output_path = "output_videos/person_a_demo.avi"
    save_video(annotated_frames, output_path)

    print(f"\n[DONE] Demo video generated successfully!")
    print(f"  -> File: {os.path.abspath(output_path)}")
    print(f"  -> Duration: {len(frames) / 24:.1f} seconds (24 FPS)")
    print("=" * 65)


if __name__ == "__main__":
    main()
