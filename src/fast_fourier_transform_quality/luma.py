"""Convert input arrays to the one format every metric uses."""

from __future__ import annotations

import numpy as np

_BT601 = np.array([0.299, 0.587, 0.114])


def as_luma(video: np.ndarray) -> np.ndarray:
    """Return a float64 luma video of shape (frames, height, width).

    Accepts (T, H, W) or RGB (T, H, W, 3). Integer input is scaled to [0, 1];
    float input is assumed to be in that range already.
    """
    array = np.asarray(video)
    if array.ndim == 4:
        if array.shape[-1] != 3:
            raise ValueError(f"expected 3 colour channels, got shape {array.shape}")
    elif array.ndim != 3:
        raise ValueError(f"expected (T, H, W) or (T, H, W, 3), got shape {array.shape}")

    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(np.float64)

    if array.ndim == 4:
        array = array @ _BT601
    if not np.isfinite(array).all():
        raise ValueError("video contains NaN or infinite values")
    return array


def image_as_luma(image: np.ndarray) -> np.ndarray:
    """Return a float64 luma image of shape (height, width), from (H, W) or RGB (H, W, 3)."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] != 3:
        raise ValueError(f"expected 3 colour channels, got shape {array.shape}")
    if array.ndim not in (2, 3):
        raise ValueError(f"expected (H, W) or (H, W, 3), got shape {array.shape}")
    return as_luma(array[np.newaxis])[0]
