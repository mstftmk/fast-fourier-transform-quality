"""Regenerate the two figures at the top of the README: docs/fft_rebuild.gif and docs/fft_hero.png.

Needs: pip install -e ".[examples]". The photo is scikit-image's "coffee" (CC0, Rachel Michetti,
courtesy of Pikolo Espresso Bar).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.signal import get_window  # noqa: E402

from photo_example import degradations, psnr  # noqa: E402

DOCS = Path(__file__).parents[1] / "docs"
SIZE = 256


def center_crop(image: np.ndarray) -> np.ndarray:
    top, left = (image.shape[0] - SIZE) // 2, (image.shape[1] - SIZE) // 2
    return image[top : top + SIZE, left : left + SIZE]


def log_spectrum(image: np.ndarray) -> np.ndarray:
    """Centered log power spectrum, with a Hann window to suppress edge leakage."""
    window = np.outer(get_window("hann", SIZE), get_window("hann", SIZE))
    spectrum = np.fft.fftshift(np.fft.fft2((image - image.mean()) * window))
    return np.log10(np.abs(spectrum) ** 2 + 1e-6)


def rebuild_gif(reference: np.ndarray) -> None:
    """Rebuild the photo from a growing disk of frequencies, coarse to fine."""
    spectrum = np.fft.fftshift(np.fft.fft2(reference))
    display = np.log10(np.abs(spectrum) ** 2 + 1)
    fy, fx = np.meshgrid(np.fft.fftfreq(SIZE), np.fft.fftfreq(SIZE), indexing="ij")
    radius = np.fft.fftshift(np.hypot(fy, fx))

    fig, (photo, panel) = plt.subplots(1, 2, figsize=(7.0, 3.8), dpi=80)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03, wspace=0.05)
    frames = []
    cutoffs = np.geomspace(1.5 / SIZE, np.sqrt(0.5), 30)
    for cutoff in cutoffs:
        mask = radius <= cutoff
        rebuilt = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
        photo.clear()
        panel.clear()
        photo.imshow(rebuilt, cmap="gray", vmin=0, vmax=1)
        photo.set_title("Photo rebuilt from these waves", fontsize=10)
        panel.imshow(np.where(mask, display, display.min()), cmap="magma")
        share = 100 * mask.mean()
        label = f"{share:.2f}" if share < 1 else f"{share:.0f}"
        panel.set_title(f"FFT: {label}% of the frequencies used", fontsize=10)
        for axis in (photo, panel):
            axis.set_xticks([])
            axis.set_yticks([])
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(pixels).quantize(colors=128, dither=Image.Dither.NONE))
    plt.close(fig)

    durations = [140] * (len(frames) - 1) + [1800]
    frames[0].save(
        DOCS / "fft_rebuild.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def hero_figure(reference: np.ndarray, cases: dict[str, np.ndarray]) -> None:
    blurred, compressed, noisy = cases.values()
    panels = {"Original": reference, "Blur": blurred, "JPEG": compressed, "Noise": noisy}
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.5))
    for column, (name, image) in enumerate(panels.items()):
        crop = center_crop(image)
        axes[0, column].imshow(crop, cmap="gray", vmin=0, vmax=1)
        title = "Original" if name == "Original" else f"{name}: PSNR {psnr(image, reference):.1f} dB"
        axes[0, column].set_title(title, fontsize=11)
        axes[1, column].imshow(log_spectrum(crop), cmap="magma", vmin=-2, vmax=5)
    axes[0, 0].set_ylabel("Photo", fontsize=11)
    axes[1, 0].set_ylabel("FFT power spectrum", fontsize=11)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(DOCS / "fft_hero.png", dpi=110)
    plt.close(fig)


def main() -> None:
    reference, cases = degradations()
    hero_figure(reference, cases)
    rebuild_gif(center_crop(reference))
    print("wrote", DOCS / "fft_hero.png", "and", DOCS / "fft_rebuild.gif")


if __name__ == "__main__":
    main()
