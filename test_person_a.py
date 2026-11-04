"""
Test suite for Person A's deliverables:
  - Tracker class (trackers/tracker.py)
  - CameraMovementEstimator (camera_movement_estimator/camera_movement_estimator.py)
  - Utility functions (utils/bbox_utils.py)

Runs without a trained YOLO model by using dummy stubs and synthetic frames.
"""

import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def test_bbox_utils():
    """Test all bounding box utility functions."""
    from utils import get_center_of_bbox, get_bbox_width, get_foot_position, measure_distance

    print("Testing bbox_utils...")

    # get_center_of_bbox
    bbox = [100, 200, 300, 400]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 200 and cy == 300, f"Expected (200, 300), got ({cx}, {cy})"

    # get_bbox_width
    width = get_bbox_width(bbox)
    assert width == 200, f"Expected 200, got {width}"

    # get_foot_position
    fx, fy = get_foot_position(bbox)
    assert fx == 200 and fy == 400, f"Expected (200, 400), got ({fx}, {fy})"

    # measure_distance
    dist = measure_distance((0, 0), (3, 4))
    assert abs(dist - 5.0) < 1e-9, f"Expected 5.0, got {dist}"

    dist2 = measure_distance((10, 10), (10, 10))
    assert dist2 == 0.0, f"Expected 0.0, got {dist2}"

    print("  [PASS] All bbox_utils tests passed")


def test_dummy_stub_generation():
    """Test that generate_dummy_stub creates valid tracks."""
    from generate_dummy_stub import generate_dummy_tracks

    print("Testing dummy stub generation...")

    tracks = generate_dummy_tracks(num_frames=50, num_players=10, num_referees=2)

    assert "players" in tracks, "Missing 'players' key"
    assert "referees" in tracks, "Missing 'referees' key"
    assert "ball" in tracks, "Missing 'ball' key"

    assert len(tracks["players"]) == 50, f"Expected 50 frames, got {len(tracks['players'])}"
    assert len(tracks["referees"]) == 50
    assert len(tracks["ball"]) == 50

    # Check first frame has expected player count
    assert len(tracks["players"][0]) == 10, f"Expected 10 players, got {len(tracks['players'][0])}"
    assert len(tracks["referees"][0]) == 2

    # Check bbox format
    first_player = list(tracks["players"][0].values())[0]
    assert "bbox" in first_player, "Missing 'bbox' key in player track"
    assert len(first_player["bbox"]) == 4, "bbox should have 4 values"

    # Check ball has some missing frames (for interpolation testing)
    ball_detected = sum(1 for b in tracks["ball"] if len(b) > 0)
    ball_missing = 50 - ball_detected
    assert ball_missing > 0, "Expected some missing ball frames for interpolation test"

    print(f"  âœ“ Stub generated: {ball_detected}/50 ball detections ({ball_missing} missing)")


def test_stub_pickle_roundtrip():
    """Test that tracks can be saved/loaded via pickle."""
    from generate_dummy_stub import generate_dummy_tracks

    print("Testing pickle stub roundtrip...")

    tracks = generate_dummy_tracks(num_frames=10)
    stub_path = os.path.join("stubs", "test_stub.pkl")
    os.makedirs("stubs", exist_ok=True)

    with open(stub_path, "wb") as f:
        pickle.dump(tracks, f)

    with open(stub_path, "rb") as f:
        loaded = pickle.load(f)

    assert loaded["players"][0] == tracks["players"][0], "Pickle roundtrip mismatch"
    assert len(loaded["ball"]) == len(tracks["ball"])

    # Cleanup
    os.remove(stub_path)

    print("  âœ“ Pickle roundtrip passed")


def test_ball_interpolation():
    """Test ball interpolation fills missing frames."""
    print("Testing ball interpolation...")

    # We can't instantiate Tracker without a YOLO model, so test the
    # interpolation logic directly via pandas
    import pandas as pd

    # Simulate ball positions with gaps
    ball_positions = [
        {1: {"bbox": [100, 100, 120, 120]}},  # Frame 0
        {},                                      # Frame 1 - missing
        {},                                      # Frame 2 - missing
        {1: {"bbox": [160, 160, 180, 180]}},  # Frame 3
        {1: {"bbox": [180, 180, 200, 200]}},  # Frame 4
    ]

    # Extract raw positions (same logic as Tracker.interpolate_ball_positions)
    raw = [
        x.get(1, {}).get("bbox", [np.nan, np.nan, np.nan, np.nan])
        if len(x.get(1, {}).get("bbox", [])) == 4
        else [np.nan, np.nan, np.nan, np.nan]
        for x in ball_positions
    ]
    df = pd.DataFrame(raw, columns=["x1", "y1", "x2", "y2"])
    df = df.interpolate()
    df = df.bfill()

    result = [{1: {"bbox": x}} for x in df.to_numpy().tolist()]

    # Check no NaN values remain
    for i, frame in enumerate(result):
        bbox = frame[1]["bbox"]
        assert all(not np.isnan(v) for v in bbox), f"NaN found in frame {i}"

    # Check interpolated values are between endpoints
    frame1_bbox = result[1][1]["bbox"]
    assert 100 < frame1_bbox[0] < 160, f"Interpolation wrong: x1={frame1_bbox[0]}"

    print(f"  âœ“ Interpolation filled {2} missing frames correctly")


def test_camera_movement_estimator():
    """Test CameraMovementEstimator with synthetic frames."""
    from camera_movement_estimator import CameraMovementEstimator

    print("Testing CameraMovementEstimator...")

    # Create synthetic frames (640x480 with a gradient pattern)
    height, width = 480, 640
    frames = []

    for i in range(5):
        # Create a frame with horizontal gradient shifted by i*10 pixels
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        shift = i * 10

        # Add textured pattern in banner regions for feature detection
        for y in range(height):
            for x in range(0, width, 20):
                val = ((x + shift) * 7 + y * 3) % 256
                x_end = min(x + 20, width)
                frame[y, x:x_end] = [val, (val + 80) % 256, (val + 160) % 256]

        frames.append(frame)

    # Initialize estimator with first frame
    estimator = CameraMovementEstimator(frames[0])

    # Verify mask was created correctly
    mask = estimator.features["mask"]
    assert mask.shape == (height, width), f"Mask shape mismatch: {mask.shape}"

    # Top 20% should be 255
    top_region = mask[0:int(height * 0.2), :]
    assert np.all(top_region == 255), "Top banner mask incorrect"

    # Center should be 0
    center_region = mask[int(height * 0.3):int(height * 0.7), :]
    assert np.all(center_region == 0), "Center mask should exclude players"

    # Bottom 20% should be 255
    bottom_region = mask[int(height * 0.8):, :]
    assert np.all(bottom_region == 255), "Bottom banner mask incorrect"

    print("  âœ“ Banner mask created correctly (top 20% + bottom 20%)")

    # Test stub save/load
    os.makedirs("stubs", exist_ok=True)
    stub_path = "stubs/test_camera_stub.pkl"

    movement = estimator.get_camera_movement(frames, stub_path=stub_path)

    assert len(movement) == len(frames), f"Expected {len(frames)} entries, got {len(movement)}"
    assert movement[0] == [0, 0], "First frame should have zero movement"

    # Test stub loading
    movement_loaded = estimator.get_camera_movement(
        frames, read_from_stub=True, stub_path=stub_path
    )
    assert movement_loaded == movement, "Stub load mismatch"

    # Cleanup
    os.remove(stub_path)

    print(f"  âœ“ Camera movement estimated for {len(frames)} frames")
    print(f"  âœ“ Stub save/load roundtrip passed")


def test_position_adjustment():
    """Test that camera movement is correctly subtracted from positions."""
    from camera_movement_estimator import CameraMovementEstimator

    print("Testing position adjustment...")

    # Create a dummy frame for initialization
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    estimator = CameraMovementEstimator(frame)

    # Create mock tracks with positions
    tracks = {
        "players": [
            {1: {"bbox": [100, 100, 150, 200], "position": (125, 200)}},
            {1: {"bbox": [110, 105, 160, 205], "position": (135, 205)}},
        ],
        "ball": [
            {1: {"bbox": [200, 200, 220, 220], "position": (210, 210)}},
            {1: {"bbox": [215, 205, 235, 225], "position": (225, 215)}},
        ],
        "referees": [
            {1: {"bbox": [300, 300, 340, 400], "position": (320, 400)}},
            {1: {"bbox": [305, 305, 345, 405], "position": (325, 405)}},
        ],
    }

    camera_movement = [[0, 0], [10, 5]]

    estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    # Frame 0: no camera movement, position_adjusted == position
    adj = tracks["players"][0][1]["position_adjusted"]
    assert adj == (125, 200), f"Frame 0 adjustment wrong: {adj}"

    # Frame 1: camera moved [10, 5], so subtract
    adj = tracks["players"][1][1]["position_adjusted"]
    assert adj == (125, 200), f"Frame 1 adjustment wrong: {adj}"  # 135-10, 205-5

    adj_ball = tracks["ball"][1][1]["position_adjusted"]
    assert adj_ball == (215, 210), f"Ball adjustment wrong: {adj_ball}"  # 225-10, 215-5

    print("  âœ“ Position adjustment correctly subtracts camera movement")


def main():
    print("=" * 60)
    print("  Person A Test Suite â€” Tracker & Camera Estimator")
    print("=" * 60)
    print()

    tests = [
        test_bbox_utils,
        test_dummy_stub_generation,
        test_stub_pickle_roundtrip,
        test_ball_interpolation,
        test_camera_movement_estimator,
        test_position_adjustment,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  âœ— FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

