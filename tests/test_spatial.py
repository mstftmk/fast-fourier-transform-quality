import numpy as np
import pytest
from scipy.signal import get_window

from fast_fourier_transform_quality.spatial import (
    DEFAULT_BANDS,
    band_energies,
    radial_power_spectrum,
    video_radial_spectrum,
)
from fast_fourier_transform_quality.synthetic import add_blur, textured_video


def horizontal_wave(cycles, height=128, width=128):
    y = np.arange(height)[:, None]
    return np.tile(np.sin(2 * np.pi * cycles * y / height), (1, width))


@pytest.mark.parametrize("cycles", [4, 16, 40])
def test_pure_wave_peaks_at_its_frequency(cycles):
    spectrum = radial_power_spectrum(horizontal_wave(cycles))
    assert spectrum.frequencies[np.argmax(spectrum.power)] == pytest.approx(cycles / 128)


def test_non_square_frame_keeps_frequencies_in_cycles_per_pixel():
    spectrum = radial_power_spectrum(horizontal_wave(16, height=64, width=128))
    assert spectrum.frequencies[np.argmax(spectrum.power)] == pytest.approx(16 / 64)


def test_power_sums_to_windowed_variance():
    frame = horizontal_wave(8) + 0.5 * horizontal_wave(20).T
    window = np.outer(get_window("hann", 128), get_window("hann", 128))
    expected = np.sum(((frame - frame.mean()) * window) ** 2) / np.sum(window**2)
    assert radial_power_spectrum(frame).power.sum() == pytest.approx(expected, rel=1e-6)


def test_brightness_has_no_effect_and_contrast_scales_quadratically():
    frame = textured_video(n_frames=1)[0]
    base = radial_power_spectrum(frame).power
    np.testing.assert_allclose(radial_power_spectrum(frame + 0.2).power, base, atol=1e-12)
    np.testing.assert_allclose(radial_power_spectrum(2 * frame).power, 4 * base, atol=1e-12)


def test_bands_partition_the_spectrum():
    spectrum = video_radial_spectrum(textured_video(n_frames=2))
    assert sum(band_energies(spectrum).values()) == pytest.approx(spectrum.power.sum())


def test_stronger_blur_removes_more_high_frequency_energy():
    video = textured_video(n_frames=4)
    high = [
        band_energies(video_radial_spectrum(add_blur(video, sigma)))["high"]
        for sigma in (0.0, 0.5, 1.0, 2.0)
    ]
    assert high == sorted(high, reverse=True)
    assert high[-1] < 0.01 * high[0]


def test_max_frames_samples_evenly_including_both_ends():
    video = np.stack([horizontal_wave(4), horizontal_wave(16), horizontal_wave(40)])
    sampled = video_radial_spectrum(video, max_frames=2)
    expected = (radial_power_spectrum(video[0]).power + radial_power_spectrum(video[2]).power) / 2
    np.testing.assert_allclose(sampled.power, expected)


def test_custom_bands_and_invalid_bands():
    spectrum = video_radial_spectrum(textured_video(n_frames=1))
    assert set(band_energies(spectrum, {"all": (0.0, 0.5)})) == {"all"}
    assert set(band_energies(spectrum)) == set(DEFAULT_BANDS)
    with pytest.raises(ValueError):
        band_energies(spectrum, {"bad": (0.3, 0.2)})


@pytest.mark.parametrize("shape", [(32,), (4, 32), (8, 8, 8)])
def test_rejects_wrong_frame_shapes(shape):
    with pytest.raises(ValueError):
        radial_power_spectrum(np.zeros(shape))


def test_rejects_invalid_max_frames():
    with pytest.raises(ValueError):
        video_radial_spectrum(textured_video(n_frames=1), max_frames=0)
