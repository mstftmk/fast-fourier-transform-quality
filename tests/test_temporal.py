import numpy as np
import pytest

from fast_fourier_transform_quality.synthetic import add_flicker, textured_video
from fast_fourier_transform_quality.temporal import temporal_bands, temporal_spectrum

FPS = 24.0


def test_oscillating_pixels_peak_at_their_frequency():
    time = np.arange(128) / FPS
    video = np.sin(2 * np.pi * 6.0 * time)[:, None, None] * np.ones((128, 16, 16))
    spectrum = temporal_spectrum(video, FPS)
    assert spectrum.frequencies[np.argmax(spectrum.density)] == pytest.approx(6.0)


def test_density_integrates_to_temporal_variance():
    rng = np.random.default_rng(0)
    video = 0.5 + 0.1 * rng.standard_normal((256, 32, 32))
    spectrum = temporal_spectrum(video, FPS)
    energy = temporal_bands(spectrum)
    assert energy["slow"] + energy["fast"] == pytest.approx(0.01, rel=0.05)


def test_static_video_has_no_temporal_energy():
    bands = temporal_bands(temporal_spectrum(np.full((32, 16, 16), 0.5), FPS))
    assert bands["slow"] == pytest.approx(0, abs=1e-12)
    assert bands["fast"] == pytest.approx(0, abs=1e-12)


def test_cutoff_splits_energy_between_bands():
    time = np.arange(128) / FPS
    video = np.sin(2 * np.pi * 3.0 * time)[:, None, None] * np.ones((128, 16, 16))
    spectrum = temporal_spectrum(video, FPS)
    assert temporal_bands(spectrum, cutoff_hz=6.0)["fast"] < 1e-3
    assert temporal_bands(spectrum, cutoff_hz=1.0)["slow"] < 1e-3


def test_cutoff_sits_exactly_where_requested():
    time = np.arange(128) / FPS
    video = np.sin(2 * np.pi * 6.0 * time)[:, None, None] * np.ones((128, 16, 16))
    spectrum = temporal_spectrum(video, FPS)
    above = temporal_bands(spectrum, cutoff_hz=5.0)
    below = temporal_bands(spectrum, cutoff_hz=7.0)
    assert above["slow"] < 1e-3 * above["fast"]
    assert below["fast"] < 1e-3 * below["slow"]


def test_flicker_out_of_phase_across_the_frame_is_still_measured():
    time = np.arange(128) / FPS
    in_phase = np.sin(2 * np.pi * 6.0 * time)[:, None, None] * np.ones((128, 16, 16))
    phases = np.random.default_rng(0).uniform(0, 2 * np.pi, (16, 16))
    out_of_phase = np.sin(2 * np.pi * 6.0 * time[:, None, None] + phases)

    assert abs(out_of_phase.mean(axis=(1, 2))).max() < abs(in_phase.mean(axis=(1, 2))).max()
    in_energy = temporal_bands(temporal_spectrum(in_phase, FPS))["fast"]
    out_energy = temporal_bands(temporal_spectrum(out_of_phase, FPS))["fast"]
    assert out_energy == pytest.approx(in_energy, rel=0.02)


def test_stronger_flicker_adds_more_fast_energy():
    video = textured_video(n_frames=96, height=48, width=48)
    fast = [
        temporal_bands(temporal_spectrum(add_flicker(video, a, 10.0, FPS), FPS))["fast"]
        for a in (0.0, 0.01, 0.03, 0.06)
    ]
    assert fast == sorted(fast)
    assert fast[-1] > 1.5 * fast[0]


def test_spatial_stride_gives_a_close_estimate():
    video = textured_video(n_frames=96, height=64, width=64)
    full = temporal_bands(temporal_spectrum(video, FPS))["fast"]
    strided = temporal_bands(temporal_spectrum(video, FPS, spatial_stride=2))["fast"]
    assert strided == pytest.approx(full, rel=0.1)


@pytest.mark.parametrize(
    "kwargs", [{"fps": 0.0}, {"fps": -24.0}, {"fps": 24.0, "spatial_stride": 0}]
)
def test_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        temporal_spectrum(np.zeros((16, 8, 8)), **kwargs)


def test_rejects_short_clips_and_invalid_cutoff():
    with pytest.raises(ValueError):
        temporal_spectrum(np.zeros((4, 8, 8)), FPS)
    spectrum = temporal_spectrum(np.zeros((16, 8, 8)), FPS)
    with pytest.raises(ValueError):
        temporal_bands(spectrum, cutoff_hz=0)
