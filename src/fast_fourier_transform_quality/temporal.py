"""Per-pixel temporal spectra: how fast pixel values change over time.

Flicker adds energy at high temporal frequencies while single-frame metrics stay
unchanged. Each pixel's time series gets its own Welch estimate; the results are
then averaged, so flicker that is out of phase across the frame still counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from .luma import as_luma

DEFAULT_CUTOFF_HZ = 5.0


@dataclass(frozen=True)
class TemporalSpectrum:
    """Mean power spectral density over pixels; its integral is the temporal variance."""

    frequencies: np.ndarray
    density: np.ndarray


def temporal_spectrum(
    video: np.ndarray,
    fps: float,
    segment_length: int = 64,
    spatial_stride: int = 1,
) -> TemporalSpectrum:
    """Estimate the temporal spectrum with Hann-windowed Welch segments."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if spatial_stride < 1:
        raise ValueError("spatial_stride must be at least 1")
    luma = as_luma(video)[:, ::spatial_stride, ::spatial_stride]
    n_frames, height, width = luma.shape
    if n_frames < 8:
        raise ValueError(f"need at least 8 frames, got {n_frames}")

    nperseg = min(segment_length, n_frames)
    frequencies = None
    total = None
    # Rows are processed in blocks so a long video does not need one huge FFT allocation.
    for start in range(0, height, 64):
        block = luma[:, start : start + 64, :]
        frequencies, density = welch(
            block,
            fs=fps,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            axis=0,
        )
        block_sum = density.sum(axis=(1, 2))
        total = block_sum if total is None else total + block_sum
    return TemporalSpectrum(frequencies=frequencies, density=total / (height * width))


def temporal_bands(
    spectrum: TemporalSpectrum,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
) -> dict[str, float]:
    """Split temporal energy into slow (below cutoff) and fast (at or above cutoff)."""
    # Limitations:
    # - A fast value below the reference does not prove there is no flicker. Blur removes the
    #   fine texture whose motion causes fast change, which can hide flicker in this ratio.
    #   Read these bands next to the spatial bands and check the spectrum for a peak.
    # - The default cutoff is a starting point, not a calibrated value.
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive")
    step = spectrum.frequencies[1] - spectrum.frequencies[0]
    energy = spectrum.density * step
    fast = spectrum.frequencies >= cutoff_hz
    return {"slow": float(energy[~fast].sum()), "fast": float(energy[fast].sum())}
