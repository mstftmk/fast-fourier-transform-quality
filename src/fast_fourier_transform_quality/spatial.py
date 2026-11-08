"""Radially averaged power spectra: how much energy each spatial scale holds.

Blur and over-smoothing remove energy from high spatial frequencies, often with
little effect on pixel-wise metrics such as PSNR. Frequencies are in cycles per
pixel, so 0.5 is the Nyquist limit along an axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window

from .luma import as_luma

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "low": (0.0, 0.10),
    "mid": (0.10, 0.25),
    "high": (0.25, 0.50),
}


@dataclass(frozen=True)
class RadialSpectrum:
    """Power summed over each frequency ring; rings beyond Nyquist are dropped."""

    frequencies: np.ndarray
    power: np.ndarray


def radial_power_spectrum(frame: np.ndarray) -> RadialSpectrum:
    """Compute the radial power spectrum of one grayscale frame.

    The frame mean is removed, so a change in overall brightness has no effect.
    A Hann window suppresses leakage from the frame edges. Power is normalized
    so that the sum over all 2D frequencies equals the windowed variance.
    """
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError(f"expected a 2D frame, got shape {frame.shape}")
    height, width = frame.shape
    if min(height, width) < 8:
        raise ValueError(f"frame too small for a spectrum: {frame.shape}")

    window = np.outer(get_window("hann", height), get_window("hann", width))
    windowed = (frame - frame.mean()) * window
    power_2d = np.abs(np.fft.fft2(windowed)) ** 2 / (height * width * np.sum(window**2))

    radius = np.hypot(*np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width), indexing="ij"))
    bin_width = 1.0 / min(height, width)
    n_bins = int(round(0.5 / bin_width)) + 1
    bins = np.rint(radius / bin_width).astype(np.int64)
    keep = bins < n_bins
    ring_power = np.bincount(bins[keep], weights=power_2d[keep], minlength=n_bins)
    return RadialSpectrum(frequencies=np.arange(n_bins) * bin_width, power=ring_power)


def video_radial_spectrum(video: np.ndarray, max_frames: int | None = None) -> RadialSpectrum:
    """Average the radial spectrum over evenly spaced frames of a video."""
    luma = as_luma(video)
    n_frames = luma.shape[0]
    if n_frames == 0:
        raise ValueError("video has no frames")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    count = n_frames if max_frames is None else min(max_frames, n_frames)
    indices = np.linspace(0, n_frames - 1, count).round().astype(int)

    spectra = [radial_power_spectrum(luma[i]) for i in indices]
    return RadialSpectrum(
        frequencies=spectra[0].frequencies,
        power=np.mean([s.power for s in spectra], axis=0),
    )


def band_energies(
    spectrum: RadialSpectrum,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Sum ring power inside each [low, high) band; the top edge of the spectrum is included."""
    # Limitations:
    # - Spectra describe statistics, not content. A different scene with similar statistics
    #   scores close to 1, so a good result does not mean the right subject, identity or action.
    #   Pair these bands with content-aware metrics.
    # - More energy is not always better. Noise raises the high band without adding detail,
    #   so check that a fix looks right before trusting the number.
    # - Not every visible defect moves the bands. JPEG blocks are obvious to the eye but change
    #   the bands only moderately, because rings average over the whole image and all angles.
    # - The default bands are starting points, not calibrated values.
    bands = DEFAULT_BANDS if bands is None else bands
    top = spectrum.frequencies[-1]
    energies = {}
    for name, (low, high) in bands.items():
        if not 0 <= low < high:
            raise ValueError(f"invalid band {name}: ({low}, {high})")
        upper = spectrum.frequencies <= top if high >= top else spectrum.frequencies < high
        energies[name] = float(spectrum.power[(spectrum.frequencies >= low) & upper].sum())
    return energies
