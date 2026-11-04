"""
Video I/O utilities.

Provides read_video, get_video_fps, and save_video.
Keeps a stable public API so that Person A's tracker and main.py can
import these functions without change.
"""

import os

import cv2


def read_video(path):
    """
    Load a video file into a list of BGR frame arrays.

    Args:
        path (str | os.PathLike): Path to the input video file.

    Returns:
        list[np.ndarray]: List of BGR frames in display order.

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
        IOError: If OpenCV cannot open the file or the file has zero frames.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path!r}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {path!r}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise IOError(f"Video opened but contains zero readable frames: {path!r}")

    return frames


def get_video_fps(path):
    """
    Read the FPS metadata of a video file.

    Falls back to 24.0 when OpenCV reports 0 (e.g. for some AVI containers
    without a valid frame-rate header).

    Args:
        path (str | os.PathLike): Path to a video file.

    Returns:
        float: Frames per second (always > 0).

    # TODO(A-PUSH): A's speed estimator should call get_video_fps(video_path)
    #   instead of hard-coding 24 fps so that slow-motion or high-frame-rate
    #   footage produces correct km/h values.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 0:
        return 24.0
    return float(fps)


def save_video(frames, path, fps=24):
    """
    Write a list of BGR frames to an XVID .avi file.

    The output directory is created automatically if it does not already exist.
    Frame dimensions are taken from the first frame.

    Args:
        frames (list[np.ndarray]): BGR frames to write.
        path (str | os.PathLike): Destination file path (should end in .avi).
        fps (int | float): Output frame rate. Defaults to 24.

    Raises:
        ValueError: If *frames* is empty.
    """
    if not frames:
        raise ValueError("Cannot save an empty frame list.")

    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
