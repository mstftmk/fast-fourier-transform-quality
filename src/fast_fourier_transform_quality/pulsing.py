"""Keyframe pulsing: a periodic jump in the picture, as left by video compression.

Encoders insert a keyframe at a regular interval, and its quality often differs from the
frames before it. The change between consecutive frames then spikes once per interval.
The period is found from the autocorrelation of the frame change signal, computed with
the FFT (the Wiener-Khinchin theorem: autocorrelation is the inverse FFT of the power
spectrum). This is a practical heuristic, not a standard metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .luma import as_luma

# A shorter lag within this share of the strongest peak wins, because a repeating
# pattern also correlates at every multiple of its period.
_PEAK_TOLERANCE = 0.85


def frame_changes(video: np.ndarray) -> np.ndarray:
    """RMS change between frames; element i is the step from frame i to frame i + 1."""
    luma = as_luma(video)
    if luma.shape[0] < 2:
        raise ValueError("need at least 2 frames")
    return np.sqrt(np.mean(np.diff(luma, axis=0) ** 2, axis=(1, 2)))


def autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation by lag, from the FFT, corrected for the shrinking overlap."""
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size
    centered = signal - signal.mean()
    # Padding to 2n gives linear rather than circular correlation.
    power = np.abs(np.fft.rfft(centered, 2 * n)) ** 2
    result = np.fft.irfft(power)[:n]
    if result[0] <= 1e-12 * n * max(signal.mean() ** 2, 1e-12):
        return np.zeros(n)
    return result / result[0] * n / (n - np.arange(n))


@dataclass(frozen=True)
class PulsingEstimate:
    """A repeating pattern in the frame changes.

    period: frames between spikes.
    strength: how far the largest position in the cycle rises above the median position.
        0.5 means spikes 1.5 times the typical change.
    autocorrelation: correlation of the signal with itself at that period.
    """

    period: int
    strength: float
    autocorrelation: float


def find_keyframe_pulsing(
    video: np.ndarray,
    min_period: int = 6,
    max_period: int = 64,
    min_cycles: int = 4,
    min_strength: float = 0.15,
) -> PulsingEstimate | None:
    """Return the repeating spike pattern in the frame changes, or None.

    Periods below min_period are skipped because B-frame patterns repeat at short
    lags. Periods with fewer than min_cycles full cycles are skipped because they
    have too few samples. min_strength is a screening value from x264 trials, not a
    statistical test.
    """
    # Limitations:
    # - This can raise false alarms, because it responds to any repeating change between frames.
    #   Flicker on a blurred clip can report a period, so read it with the temporal spectrum.
    # - Weak compression may fall below min_strength.
    # - Patterns shorter than min_period can be reported as a multiple of their period.
    # - An interval needs at least min_cycles full cycles in the clip.
    # - Real encoders can vary their keyframe interval, which weakens the pattern.
    # - Tested on x264 encodes of synthetic pans, not on camera footage or model output.
    if min_period < 2:
        raise ValueError("min_period must be at least 2")
    if min_cycles < 1:
        raise ValueError("min_cycles must be at least 1")
    changes = frame_changes(video)
    correlation = autocorrelation(changes)
    last = min(max_period, changes.size // min_cycles)
    peaks = [
        lag
        for lag in range(min_period, last + 1)
        if lag + 1 < correlation.size
        and correlation[lag] > correlation[lag - 1]
        and correlation[lag] >= correlation[lag + 1]
    ]
    if not peaks:
        return None
    strongest = max(correlation[lag] for lag in peaks)
    if strongest <= 0:
        return None
    period = next(lag for lag in peaks if correlation[lag] >= _PEAK_TOLERANCE * strongest)

    positions = np.arange(changes.size) % period
    means = np.array([changes[positions == i].mean() for i in range(period)])
    median = np.median(means)
    if median <= 0:
        return None
    strength = float(means.max() / median - 1)
    if strength < min_strength:
        return None
    return PulsingEstimate(
        period=int(period), strength=strength, autocorrelation=float(correlation[period])
    )
