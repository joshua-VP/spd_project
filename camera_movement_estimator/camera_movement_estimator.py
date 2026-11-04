import pickle
import cv2
import numpy as np
import os
import sys

sys.path.append("../")
from utils import measure_distance


class CameraMovementEstimator:
    """
    Estimates camera panning movement between consecutive video frames
    using Lucas-Kanade Optical Flow.

    Uses feature points detected in static stadium advertising banner regions
    (top and bottom of frame) to isolate camera motion from player movement.
    """

    def __init__(self, frame):
        """
        Initialize the camera movement estimator.

        Sets up Lucas-Kanade optical flow parameters and corner detection
        parameters. Creates a mask that restricts feature detection to
        stadium banner regions (top and bottom strips), excluding the
        central pitch area where players move.

        Args:
            frame (np.ndarray): First video frame, used to determine dimensions
                and initialize the feature detection mask.
        """
        self.minimum_distance = 5  # Pixel threshold to filter noise

        # Lucas-Kanade Optical Flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        # Shi-Tomasi corner detection parameters
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=None,
        )

        # Create mask for stadium banner regions only
        # Top banner: 0 to 20% of frame height
        # Bottom banner: 80% to 100% of frame height
        # This excludes the central pitch where players move
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, :] = 0

        height = first_frame_grayscale.shape[0]
        # Top strip — captures upper advertising banners
        mask_features[0 : int(height * 0.2), :] = 255
        # Bottom strip — captures lower advertising banners
        mask_features[int(height * 0.8) :, :] = 255

        self.features["mask"] = mask_features

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Subtract cumulative camera movement from tracked object positions.

        Adjusts player/referee/ball foot coordinates so that panning camera
        motion does not skew true player positions on the pitch.

        Args:
            tracks (dict): The tracks dictionary containing bounding boxes
                and positions for players, referees, and ball.
            camera_movement_per_frame (list[list[float]]): Per-frame camera
                displacement as [[dx, dy], ...].
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get("position")
                    if position is None:
                        continue
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object_type][frame_num][track_id][
                        "position_adjusted"
                    ] = position_adjusted

    def get_camera_movement(
        self, frames, read_from_stub=False, stub_path=None
    ):
        """
        Estimate camera movement across all video frames using optical flow.

        Detects feature points in stadium banner regions of frame t-1 and
        tracks their displacement in frame t using Lucas-Kanade optical flow.
        Computes the median (Δx, Δy) shift. If total movement is below
        the minimum_distance threshold (5 pixels), it is filtered as noise
        and set to (0, 0).

        Args:
            frames (list[np.ndarray]): List of video frame arrays.
            read_from_stub (bool): If True, load from cached stub file.
            stub_path (str): Path to pickle file for caching.

        Returns:
            list[list[float]]: Per-frame camera movement as [[dx, dy], ...].
                First frame always has [0, 0] displacement.
        """
        # Check for pre-computed stub
        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0, 0] for _ in range(len(frames))]

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Skip if no features were found in previous frame
            if old_features is None or len(old_features) == 0:
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
                continue

            # Calculate optical flow for existing feature points
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is not None:
                diff_x_list = []
                diff_y_list = []

                for i, (new, old) in enumerate(
                    zip(new_features, old_features)
                ):
                    if status[i]:
                        new_point = new.ravel()
                        old_point = old.ravel()

                        diff_x = new_point[0] - old_point[0]
                        diff_y = new_point[1] - old_point[1]
                        diff_x_list.append(diff_x)
                        diff_y_list.append(diff_y)

                if len(diff_x_list) > 0:
                    median_x = float(np.median(diff_x_list))
                    median_y = float(np.median(diff_y_list))
                    distance = np.sqrt(median_x**2 + median_y**2)

                    # Filter out small movements as noise
                    if distance > self.minimum_distance:
                        camera_movement[frame_num] = [
                            median_x,
                            median_y,
                        ]

            # Update for next iteration
            old_gray = frame_gray.copy()
            old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Save to stub for future fast loading
        if stub_path is not None:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir:
                os.makedirs(stub_dir, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

