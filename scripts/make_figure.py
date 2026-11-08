"""Regenerate docs/validation.png from the synthetic controls."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from fast_fourier_transform_quality.spatial import video_radial_spectrum
from fast_fourier_transform_quality.synthetic import add_blur, add_flicker, textured_video
from fast_fourier_transform_quality.temporal import temporal_spectrum

FPS = 24.0


def main() -> None:
    clean = textured_video()
    fig, (space, time) = plt.subplots(1, 2, figsize=(10, 3.6))

    for video, label in ((clean, "clean"), (add_blur(clean, 1.5), "blurred")):
        spectrum = video_radial_spectrum(video)
        space.loglog(spectrum.frequencies[1:], spectrum.power[1:] + 1e-16, label=label)
    space.set(title="Blur removes high spatial frequencies", xlabel="cycles per pixel", ylabel="power")
    space.set_ylim(bottom=1e-10)

    for video, label in ((clean, "clean"), (add_flicker(clean, 0.03, 10.0, FPS), "10 Hz flicker")):
        spectrum = temporal_spectrum(video, FPS)
        time.semilogy(spectrum.frequencies[1:], spectrum.density[1:], label=label)
    time.set(title="Flicker adds a temporal peak", xlabel="Hz", ylabel="power density")

    for axis in (space, time):
        axis.legend(frameon=False)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    output = Path(__file__).parents[1] / "docs" / "validation.png"
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
