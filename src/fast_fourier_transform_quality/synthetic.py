"""Synthetic videos with known defects, used to check that each metric responds correctly."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def textured_video(
    n_frames: int = 96,
    height: int = 128,
    width: int = 128,
    velocity: tuple[float, float] = (0.7, 0.4),
    spectral_slope: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Create a random texture with natural-image-like falloff, translating smoothly.

    Amplitude falls as 1 / f ** spectral_slope. Motion is a sub-pixel Fourier shift
    in (rows, columns) per frame, so no interpolation blur is added.
    """
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width), indexing="ij")
    radius = np.hypot(fy, fx)
    radius[0, 0] = 1.0
    spectrum = np.fft.fft2(rng.standard_normal((height, width))) / radius**spectral_slope
    spectrum[0, 0] = 0.0

    video = np.empty((n_frames, height, width))
    for t in range(n_frames):
        shift = np.exp(-2j * np.pi * t * (fy * velocity[0] + fx * velocity[1]))
        video[t] = np.fft.ifft2(spectrum * shift).real
    return 0.5 + 0.12 * video / video.std()


def add_blur(video: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur in space only, leaving time untouched."""
    return gaussian_filter(video, sigma=(0, sigma, sigma), mode="wrap")


def add_flicker(video: np.ndarray, amplitude: float, frequency_hz: float, fps: float) -> np.ndarray:
    """Add a sinusoidal brightness oscillation to the whole frame."""
    time = np.arange(video.shape[0]) / fps
    return video + amplitude * np.sin(2 * np.pi * frequency_hz * time)[:, None, None]


def add_keyframe_pulsing(video: np.ndarray, period: int, max_sigma: float) -> np.ndarray:
    """Imitate compression: detail fades after each keyframe, then returns at the next one."""
    out = np.empty_like(video)
    for t in range(video.shape[0]):
        sigma = max_sigma * (t % period) / (period - 1)
        out[t] = gaussian_filter(video[t], sigma=sigma, mode="wrap") if sigma > 0 else video[t]
    return out


def add_grain(video: np.ndarray, amplitude: float, seed: int = 0) -> np.ndarray:
    """Add independent white noise to every pixel of every frame."""
    return video + amplitude * np.random.default_rng(seed).standard_normal(video.shape)
