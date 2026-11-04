import cv2


def read_video(video_path):
    """
    Load a video file into a list of frame arrays.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        list[np.ndarray]: List of BGR frame arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=24):
    """
    Save a list of frame arrays to a video file.

    Uses XVID codec.

    Args:
        output_video_frames (list[np.ndarray]): List of BGR frame arrays.
        output_video_path (str): Path to the output video file.
        fps (float | int): Frames per second (defaults to 24).
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()

