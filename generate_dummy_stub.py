"""
Dummy Stub Generator

Generates a mock stubs/tracks_stub.pkl file containing dummy bounding boxes
for 50 frames. This allows Person B to immediately write and test K-Means
team assignment and Ball Interpolation without waiting for Person A's YOLO model.
"""

import pickle
import os
import random

random.seed(42)


def generate_dummy_tracks(num_frames=50, num_players=10, num_referees=2):
    """
    Generate a mock tracks dictionary with realistic-looking bounding boxes.

    Simulates:
    - Players from two teams with slightly different positions
    - Referees in the central area
    - A ball that occasionally disappears (for testing interpolation)

    Args:
        num_frames (int): Number of frames to generate.
        num_players (int): Number of players per frame.
        num_referees (int): Number of referees per frame.

    Returns:
        dict: Tracks dictionary in the expected format.
    """
    tracks = {"players": [], "referees": [], "ball": []}

    # Generate base positions for players (spread across the frame)
    player_base_positions = []
    for i in range(num_players):
        x = random.randint(200, 1700)
        y = random.randint(200, 900)
        player_base_positions.append((x, y))

    # Generate base positions for referees
    referee_base_positions = []
    for i in range(num_referees):
        x = random.randint(600, 1300)
        y = random.randint(300, 700)
        referee_base_positions.append((x, y))

    # Ball starts near one of the players
    ball_x, ball_y = 960, 540

    for frame_num in range(num_frames):
        # Players
        players_dict = {}
        for player_id in range(1, num_players + 1):
            base_x, base_y = player_base_positions[player_id - 1]
            # Add some movement noise
            x1 = base_x + random.randint(-20, 20)
            y1 = base_y + random.randint(-15, 15)
            w = random.randint(40, 60)
            h = random.randint(80, 120)
            x2 = x1 + w
            y2 = y1 + h
            players_dict[player_id] = {"bbox": [x1, y1, x2, y2]}

        # Referees
        referees_dict = {}
        for ref_id in range(1, num_referees + 1):
            base_x, base_y = referee_base_positions[ref_id - 1]
            x1 = base_x + random.randint(-10, 10)
            y1 = base_y + random.randint(-10, 10)
            w = random.randint(35, 50)
            h = random.randint(75, 110)
            x2 = x1 + w
            y2 = y1 + h
            referees_dict[ref_id] = {"bbox": [x1, y1, x2, y2]}

        # Ball — occasionally missing to test interpolation
        ball_dict = {}
        if random.random() > 0.2:  # 80% detection rate
            ball_x += random.randint(-30, 30)
            ball_y += random.randint(-20, 20)
            ball_x = max(50, min(1870, ball_x))
            ball_y = max(50, min(1030, ball_y))
            ball_size = random.randint(10, 20)
            ball_dict[1] = {
                "bbox": [
                    ball_x - ball_size,
                    ball_y - ball_size,
                    ball_x + ball_size,
                    ball_y + ball_size,
                ]
            }

        tracks["players"].append(players_dict)
        tracks["referees"].append(referees_dict)
        tracks["ball"].append(ball_dict)

    return tracks


if __name__ == "__main__":
    os.makedirs("stubs", exist_ok=True)
    stub_path = os.path.join("stubs", "tracks_stub.pkl")

    tracks = generate_dummy_tracks()

    with open(stub_path, "wb") as f:
        pickle.dump(tracks, f)

    print(f"Dummy stub generated at: {stub_path}")
    print(f"  Frames: {len(tracks['players'])}")
    print(f"  Players per frame: {len(tracks['players'][0])}")
    print(f"  Referees per frame: {len(tracks['referees'][0])}")

    # Report ball detection stats
    ball_detected = sum(1 for b in tracks["ball"] if len(b) > 0)
    print(f"  Ball detected in: {ball_detected}/{len(tracks['ball'])} frames")

