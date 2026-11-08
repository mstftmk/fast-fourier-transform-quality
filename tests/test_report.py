import json

import numpy as np
import pytest

from fast_fourier_transform_quality import compare_videos
from fast_fourier_transform_quality.luma import as_luma
from fast_fourier_transform_quality.synthetic import (
    add_blur,
    add_flicker,
    add_grain,
    textured_video,
)
from fast_fourier_transform_quality.validation import FPS, controls_table


@pytest.fixture(scope="module")
def clean():
    return textured_video()


def test_identical_videos_score_one(clean):
    report = compare_videos(clean, clean, FPS)
    assert all(v == pytest.approx(1.0) for v in report.spatial_ratio.values())
    assert all(v == pytest.approx(1.0) for v in report.temporal_ratio.values())


def test_blur_reduces_high_band(clean):
    report = compare_videos(add_blur(clean, 1.5), clean, FPS)
    assert report.spatial_ratio["high"] < 0.05
    assert report.spatial_ratio["high"] < report.spatial_ratio["mid"] < report.spatial_ratio["low"]


def test_global_flicker_changes_time_but_not_space(clean):
    report = compare_videos(add_flicker(clean, 0.03, 10.0, FPS), clean, FPS)
    assert report.temporal_ratio["fast"] > 1.2
    assert report.temporal_ratio["slow"] == pytest.approx(1.0, abs=0.02)
    assert all(v == pytest.approx(1.0) for v in report.spatial_ratio.values())


def test_blur_can_hide_flicker_in_the_fast_band(clean):
    # Known limitation, see temporal_bands: a fast ratio below 1 does not prove there is no flicker.
    blurred = add_blur(clean, 0.8)
    blur_only = compare_videos(blurred, clean, FPS).temporal_ratio["fast"]
    both = compare_videos(add_flicker(blurred, 0.03, 10.0, FPS), clean, FPS).temporal_ratio["fast"]
    assert both < 1.0
    assert both > 5 * blur_only


def test_grain_raises_high_frequencies_in_space_and_time(clean):
    report = compare_videos(add_grain(clean, 0.02, seed=1), clean, FPS)
    assert report.spatial_ratio["high"] > 1.05
    assert report.temporal_ratio["fast"] > 1.1


@pytest.mark.parametrize("seed", [100, 107])
def test_metrics_cannot_tell_that_the_content_is_different(clean, seed):
    # Known limitation, see band_energies: spectra describe statistics, not content.
    # A different scene with similar statistics passes.
    report = compare_videos(textured_video(seed=seed), clean, FPS)
    assert report.spatial_ratio["high"] == pytest.approx(1.0, abs=0.2)
    assert report.temporal_ratio["fast"] == pytest.approx(1.0, abs=0.2)


def test_different_frame_sizes_are_rejected(clean):
    with pytest.raises(ValueError, match="frame sizes differ"):
        compare_videos(clean, clean[:, :64, :64], FPS)


def test_report_serializes_to_json_and_text(clean):
    report = compare_videos(add_blur(clean, 1.5), clean, FPS)
    decoded = json.loads(json.dumps(report.to_dict()))
    assert decoded["spatial_ratio"]["high"] < 0.05
    assert decoded["cutoff_hz"] == 5.0
    assert "Temporal energy" in report.to_text()


def test_luma_conversion_of_rgb_and_integer_input():
    gray_rgb = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    np.testing.assert_allclose(as_luma(gray_rgb), 1.0)
    pure_green = np.zeros((1, 4, 4, 3))
    pure_green[..., 1] = 1.0
    np.testing.assert_allclose(as_luma(pure_green), 0.587)


@pytest.mark.parametrize("shape", [(4, 4), (2, 4, 4, 1)])
def test_luma_rejects_wrong_shapes(shape):
    with pytest.raises(ValueError):
        as_luma(np.zeros(shape))


def test_luma_rejects_non_finite_values():
    with pytest.raises(ValueError):
        as_luma(np.full((2, 4, 4), np.nan))


def test_controls_table_has_a_row_per_control():
    assert len(controls_table().splitlines()) == 2 + 6
