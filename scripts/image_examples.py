"""Regenerate the image examples in the README and print their tables.

Writes docs/image_gallery.png, docs/image_processing.png and docs/periodic_pattern.png.
Needs: pip install -e ".[examples]". Photos come from scikit-image: chelsea (CC0, Stefan van der
Walt), brick and grass (CC0), rocket (public domain, SpaceX).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.signal import get_window  # noqa: E402
from skimage import data  # noqa: E402

from fast_fourier_transform_quality import compare_images  # noqa: E402
from fast_fourier_transform_quality.luma import image_as_luma  # noqa: E402

DOCS = Path(__file__).parents[1] / "docs"
SIZE = 256


def psnr(test: np.ndarray, reference: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / np.mean((test - reference) ** 2)))


def center_crop(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    top, left = (image.shape[0] - size) // 2, (image.shape[1] - size) // 2
    return image[top : top + size, left : left + size]


def log_spectrum(image: np.ndarray) -> np.ndarray:
    window = np.outer(get_window("hann", image.shape[0]), get_window("hann", image.shape[1]))
    spectrum = np.fft.fftshift(np.fft.fft2((image - image.mean()) * window))
    return np.log10(np.abs(spectrum) ** 2 + 1e-6)


def show(axis, image, **kwargs) -> None:
    axis.imshow(image, **kwargs)
    axis.set_xticks([])
    axis.set_yticks([])


def photo_and_spectrum_figure(panels: dict[str, np.ndarray], path: Path, width: float) -> None:
    fig, axes = plt.subplots(2, len(panels), figsize=(width, 4.6), squeeze=False)
    for column, (name, image) in enumerate(panels.items()):
        crop = center_crop(image)
        show(axes[0, column], crop, cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(name, fontsize=10)
        show(axes[1, column], log_spectrum(crop), cmap="magma", vmin=-2, vmax=5)
    axes[0, 0].set_ylabel("Photo")
    axes[1, 0].set_ylabel("FFT power spectrum")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gallery() -> None:
    panels = {
        "Cat (natural photo)": image_as_luma(data.chelsea()),
        "Brick wall (repeating)": image_as_luma(data.brick()),
        "Grass (fine texture)": image_as_luma(data.grass()),
        "Rocket (smooth sky)": image_as_luma(data.rocket()),
    }
    photo_and_spectrum_figure(panels, DOCS / "image_gallery.png", 10.5)


def processing() -> None:
    reference_8bit = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2GRAY)
    reference = image_as_luma(reference_8bit)
    height, width = reference.shape

    small = cv2.resize(reference, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
    bicubic = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    nearest = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    sharpened = np.clip(reference + 1.5 * (reference - gaussian_filter(reference, 2.0)), 0, 1)

    noise = np.random.default_rng(0).standard_normal(reference.shape)
    noisy = np.clip(reference + 0.06 * noise, 0, 1)
    noisy_8bit = (noisy * 255).round().astype(np.uint8)
    denoised = image_as_luma(cv2.fastNlMeansDenoising(noisy_8bit, None, h=16))

    cases = {
        "4x down, bicubic up": bicubic,
        "4x down, nearest up": nearest,
        "Sharpened": sharpened,
        "Noisy": noisy,
        "Noisy, then denoised": denoised,
    }
    print("| Processing | PSNR (dB) | Low | Mid | High |")
    print("|---|---:|---:|---:|---:|")
    for name, image in cases.items():
        ratios = compare_images(image, reference).spatial_ratio
        print(
            f"| {name} | {psnr(image, reference):.1f} "
            f"| {ratios['low']:.2f} | {ratios['mid']:.2f} | {ratios['high']:.2f} |"
        )
    photo_and_spectrum_figure({"Original": reference, **cases}, DOCS / "image_processing.png", 13)


def periodic_pattern() -> None:
    reference = center_crop(image_as_luma(data.chelsea()))
    y, x = np.mgrid[:SIZE, :SIZE]
    pattern = 0.08 * np.sin(2 * np.pi * (0.11 * x + 0.07 * y))
    striped = reference + pattern

    spectrum = np.fft.fftshift(np.fft.fft2(striped - striped.mean()))
    fy, fx = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(SIZE)), np.fft.fftshift(np.fft.fftfreq(SIZE)), indexing="ij"
    )
    # Natural content is strongest at low frequencies, so search for the peak outside them.
    search = np.where(np.hypot(fy, fx) > 0.05, np.abs(spectrum), 0)
    row, column = np.unravel_index(np.argmax(search), search.shape)
    peak_y, peak_x = fy[row, column], fx[row, column]

    # A pattern is a pair of mirrored peaks, so remove both with a soft-edged notch.
    width = 3 / SIZE
    notch = (1 - np.exp(-((fy - peak_y) ** 2 + (fx - peak_x) ** 2) / (2 * width**2))) * (
        1 - np.exp(-((fy + peak_y) ** 2 + (fx + peak_x) ** 2) / (2 * width**2))
    )
    cleaned = np.fft.ifft2(np.fft.ifftshift(spectrum * notch)).real + striped.mean()

    print(f"\nPeak found at ({peak_x:.3f}, {peak_y:.3f}) cycles per pixel; true (0.110, 0.070)")
    print(f"PSNR striped {psnr(striped, reference):.1f} dB, cleaned {psnr(cleaned, reference):.1f} dB")
    for name, image in (("striped", striped), ("cleaned", cleaned)):
        ratios = compare_images(image, reference).spatial_ratio
        print(name, {key: round(value, 2) for key, value in ratios.items()})

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    show(axes[0], striped, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Photo with stripes, {psnr(striped, reference):.1f} dB", fontsize=10)
    show(axes[1], log_spectrum(striped), cmap="magma", vmin=-2, vmax=5)
    axes[1].set_title("The FFT shows two bright dots", fontsize=10)
    show(axes[2], log_spectrum(striped) * notch + (notch - 1) * 2, cmap="magma", vmin=-2, vmax=5)
    axes[2].set_title("Remove the dots", fontsize=10)
    show(axes[3], cleaned, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Back to a photo, {psnr(cleaned, reference):.1f} dB", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "periodic_pattern.png", dpi=110)
    plt.close(fig)


def main() -> None:
    gallery()
    processing()
    periodic_pattern()


if __name__ == "__main__":
    main()
