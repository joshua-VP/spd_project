"""
Generate Sample Match Video & Stub for End-to-End Pipeline Testing

Creates:
  - input_videos/input_video.mp4 (50 frames, 1920x1080, green pitch, banners, colored players)
  - stubs/tracks_stub.pkl (50 frames of matching player, referee, and ball coordinates)
"""

import os
import cv2
import pickle
import numpy as np
from generate_dummy_stub import generate_dummy_tracks


def generate_sample_match():
    os.makedirs("input_videos", exist_ok=True)
    os.makedirs("stubs", exist_ok=True)

    num_frames = 50
    width = 1920
    height = 1080

    print(f"1. Generating tracks dictionary for {num_frames} frames...")
    tracks = generate_dummy_tracks(num_frames=num_frames, num_players=10, num_referees=2)

    # Save to stub
    stub_path = "stubs/tracks_stub.pkl"
    with open(stub_path, "wb") as f:
        pickle.dump(tracks, f)
    print(f"   -> Saved tracks stub to {stub_path}")

    # Generate matching video frames
    print(f"2. Generating {num_frames} video frames ({width}x{height})...")
    frames = []

    # Team colors (Team 1: Red shirts, Team 2: Blue shirts)
    team_1_color = (40, 40, 220)   # BGR Red
    team_2_color = (220, 50, 40)   # BGR Blue
    referee_color = (40, 220, 220) # BGR Yellow
    grass_color = (40, 140, 50)    # BGR Pitch green

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = "input_videos/input_video.mp4"
    out = cv2.VideoWriter(video_path, fourcc, 24, (width, height))

    for frame_idx in range(num_frames):
        canvas = np.full((height, width, 3), grass_color, dtype=np.uint8)

        # Draw top & bottom advertising banners for optical flow
        top_h = int(height * 0.18)
        bot_h = int(height * 0.82)
        cv2.rectangle(canvas, (0, 0), (width, top_h), (80, 80, 80), -1)
        cv2.rectangle(canvas, (0, bot_h), (width, height), (80, 80, 80), -1)

        # Banner patterns
        pan = frame_idx * 5
        for x in range(-100, width + 100, 80):
            sx = (x - pan) % (width + 60)
            cv2.rectangle(canvas, (sx, 10), (sx + 40, top_h - 10), (240, 240, 240), -1)
            cv2.rectangle(canvas, (sx, bot_h + 10), (sx + 40, height - 10), (200, 200, 50), -1)

        # Field lines
        cv2.line(canvas, (width // 2, top_h), (width // 2, bot_h), (255, 255, 255), 4)
        cv2.circle(canvas, (width // 2, height // 2), 120, (255, 255, 255), 4)

        # Draw players with shirts corresponding to team colors
        for player_id, p_info in tracks["players"][frame_idx].items():
            bbox = [int(v) for v in p_info["bbox"]]
            x1, y1, x2, y2 = bbox
            shirt_color = team_1_color if player_id <= 5 else team_2_color
            # Draw top half (shirt)
            half_y = y1 + (y2 - y1) // 2
            cv2.rectangle(canvas, (x1, y1), (x2, half_y), shirt_color, -1)
            # Draw bottom half (shorts/legs)
            cv2.rectangle(canvas, (x1, half_y), (x2, y2), (240, 240, 240), -1)

        # Draw referees
        for ref_id, r_info in tracks["referees"][frame_idx].items():
            bbox = [int(v) for v in r_info["bbox"]]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), referee_color, -1)

        # Draw ball
        ball_dict = tracks["ball"][frame_idx]
        if 1 in ball_dict:
            b_box = [int(v) for v in ball_dict[1]["bbox"]]
            cx = (b_box[0] + b_box[2]) // 2
            cy = (b_box[1] + b_box[3]) // 2
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1)
            cv2.circle(canvas, (cx, cy), 8, (0, 0, 0), 1)

        out.write(canvas)

    out.release()
    print(f"   -> Saved input video to {video_path}")
    print("[SUCCESS] Sample dataset ready for testing.")


if __name__ == "__main__":
    generate_sample_match()
