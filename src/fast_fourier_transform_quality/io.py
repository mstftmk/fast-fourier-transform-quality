"""Load videos and images for the command line tool."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _import_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Reading image and video files needs OpenCV: "
            "pip install 'fast-fourier-transform-quality[video]'"
        ) from error
    return cv2


def load_image(path: str | Path) -> np.ndarray:
    """Return an image as an array: as stored for .npy, otherwise 8-bit RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    cv2 = _import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_video(path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, float | None]:
    """Return (frames, fps). A .npy file has no frame rate, so its fps is None.

    Every frame is held in memory; use max_frames for long clips.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        frames = np.load(path)
        return (frames if max_frames is None else frames[:max_frames]), None

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or None
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), fps
