"""Apply known degradations to a real photo; regenerate docs/photo_example.png and print a table.

The photo is scikit-image's "coffee" (CC0, Rachel Michetti, courtesy of Pikolo Espresso Bar).
Install the dependencies with: pip install -e ".[examples]"
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq
from skimage import data

from fast_fourier_transform_quality import compare_images, radial_power_spectrum
from fast_fourier_transform_quality.luma import image_as_luma


def psnr(test: np.ndarray, reference: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / np.mean((test - reference) ** 2)))


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def degradations() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the reference and three degradations tuned to the same PSNR."""
    reference_8bit = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2GRAY)
    reference = image_as_luma(reference_8bit)

    blurred = gaussian_filter(reference, 1.5)
    target = psnr(blurred, reference)

    # Tune the other two defects to the blur's PSNR, so that score cannot tell them apart.
    unit_noise = np.random.default_rng(0).standard_normal(reference.shape)
    amplitude = brentq(lambda a: psnr(reference + a * unit_noise, reference) - target, 1e-4, 1.0)
    noisy = reference + amplitude * unit_noise
    quality = min(
        range(1, 96),
        key=lambda q: abs(psnr(image_as_luma(jpeg(reference_8bit, q)), reference) - target),
    )
    compressed = image_as_luma(jpeg(reference_8bit, quality))
    return reference, {
        "Gaussian blur (sigma 1.5 px)": blurred,
        f"JPEG (quality {quality})": compressed,
        f"White noise (std {amplitude:.3f})": noisy,
    }


def main() -> None:
    reference, cases = degradations()
    blurred, compressed, noisy = cases.values()

    print("| Degradation | PSNR (dB) | Low | Mid | High |")
    print("|---|---:|---:|---:|---:|")
    for name, image in cases.items():
        ratios = compare_images(image, reference).spatial_ratio
        print(
            f"| {name} | {psnr(image, reference):.1f} "
            f"| {ratios['low']:.2f} | {ratios['mid']:.2f} | {ratios['high']:.2f} |"
        )

    fig = plt.figure(figsize=(13, 6))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    crop = (slice(120, 280), slice(190, 350))
    panels = {"Original": reference, "Blur": blurred, "JPEG": compressed, "Noise": noisy}
    for column, (name, image) in enumerate(panels.items()):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(image[crop], cmap="gray", vmin=0, vmax=1)
        axis.set_title(name if name == "Original" else f"{name}, {psnr(image, reference):.1f} dB")
        axis.set_xticks([])
        axis.set_yticks([])

    axis = fig.add_subplot(grid[1, :])
    for name, image in panels.items():
        spectrum = radial_power_spectrum(image)
        axis.loglog(spectrum.frequencies[1:], spectrum.power[1:], label=name)
    axis.set(xlabel="cycles per pixel", ylabel="power")
    axis.set_ylim(bottom=1e-8)
    axis.legend(frameon=False, ncol=4)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    output = Path(__file__).parents[1] / "docs" / "photo_example.png"
    fig.savefig(output, dpi=150)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
