"""Regenerate docs/fft_primer.png, the figure that introduces the Fourier transform."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fast_fourier_transform_quality.synthetic import add_blur, textured_video


def log_power_2d(frame: np.ndarray) -> np.ndarray:
    """Centered 2D power spectrum on a log scale, for display."""
    spectrum = np.fft.fftshift(np.fft.fft2(frame - frame.mean()))
    return np.log10(np.abs(spectrum) ** 2 + 1e-6)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    rate = 100
    time = np.arange(rate) / rate
    slow = np.sin(2 * np.pi * 3 * time)
    fast = 0.5 * np.sin(2 * np.pi * 10 * time)
    signal = slow + fast
    amplitude = 2 * np.abs(np.fft.rfft(signal)) / signal.size
    frequency = np.fft.rfftfreq(signal.size, 1 / rate)

    axes[0, 0].plot(time, slow, label="3 Hz")
    axes[0, 0].plot(time, fast, label="10 Hz, half as strong")
    axes[0, 0].set(title="Two pure waves", xlabel="seconds")
    axes[0, 0].legend(frameon=False, loc="upper right", ncol=2)
    axes[0, 0].set_ylim(-1.2, 1.7)
    axes[0, 1].plot(time, signal, color="black")
    axes[0, 1].set(title="Their sum: hard to read over time", xlabel="seconds")
    axes[0, 2].stem(frequency[:21], amplitude[:21], basefmt=" ")
    axes[0, 2].set(title="After the FFT: both waves are visible", xlabel="Hz", ylabel="amplitude")

    video = textured_video(n_frames=1, height=128, width=128)
    frame = video[0]
    blurred = add_blur(video, 1.5)[0]
    axes[1, 0].imshow(frame, cmap="gray", vmin=0.1, vmax=0.9)
    axes[1, 0].set(title="Synthetic frame")
    axes[1, 1].imshow(blurred, cmap="gray", vmin=0.1, vmax=0.9)
    axes[1, 1].set(title="Same frame, blurred")
    limits = {"vmin": -6, "vmax": 4, "cmap": "magma"}
    both = np.hstack([log_power_2d(frame), np.full((128, 6), -6), log_power_2d(blurred)])
    axes[1, 2].imshow(both, **limits)
    axes[1, 2].set(title="Power spectra: sharp | blurred")
    for axis in axes[1]:
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[0]:
        axis.grid(alpha=0.3)

    fig.tight_layout()
    output = Path(__file__).parents[1] / "docs" / "fft_primer.png"
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
