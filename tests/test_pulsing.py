import numpy as np
import pytest

from fast_fourier_transform_quality import find_keyframe_pulsing, frame_changes
from fast_fourier_transform_quality.pulsing import autocorrelation
from fast_fourier_transform_quality.synthetic import (
    add_blur,
    add_flicker,
    add_grain,
    add_keyframe_pulsing,
    textured_video,
)


@pytest.fixture(scope="module")
def clean():
    return textured_video(n_frames=240, height=64, width=64)


def stepped_video(n_frames=121, period=12, inside=0.01, spike=0.05):
    """Uniform frames whose brightness steps are known exactly."""
    values = np.zeros(n_frames)
    for t in range(1, n_frames):
        values[t] = values[t - 1] + (spike if t % period == 0 else inside)
    return values[:, None, None] * np.ones((n_frames, 8, 8))


def test_frame_changes_measure_rms_change():
    video = np.stack([np.zeros((4, 4)), np.full((4, 4), 0.25), np.full((4, 4), 0.25)])
    np.testing.assert_allclose(frame_changes(video), [0.25, 0.0])


def test_autocorrelation_matches_direct_computation():
    signal = np.random.default_rng(0).standard_normal(64)
    centered = signal - signal.mean()
    direct = np.correlate(centered, centered, mode="full")[63:]
    expected = direct / direct[0] * 64 / (64 - np.arange(64))
    np.testing.assert_allclose(autocorrelation(signal), expected, atol=1e-10)


def test_autocorrelation_of_a_periodic_signal_peaks_at_its_period():
    signal = np.sin(2 * np.pi * np.arange(240) / 12)
    correlation = autocorrelation(signal)
    assert correlation[12] == pytest.approx(1.0, abs=0.01)
    assert correlation[6] == pytest.approx(-1.0, abs=0.01)


def test_known_spikes_give_the_exact_period_and_strength():
    estimate = find_keyframe_pulsing(stepped_video())
    assert estimate.period == 12
    assert estimate.strength == pytest.approx(4.0)


@pytest.mark.parametrize("period", [6, 8, 12, 16, 24])
def test_finds_the_period_of_imitated_keyframes(clean, period):
    estimate = find_keyframe_pulsing(add_keyframe_pulsing(clean, period, 1.0))
    assert estimate is not None
    assert estimate.period == period


def test_finds_the_period_with_any_starting_frame(clean):
    video = add_keyframe_pulsing(clean, 12, 1.0)[5:]
    assert find_keyframe_pulsing(video).period == 12


@pytest.mark.parametrize(
    "transform",
    [lambda v: v, lambda v: add_blur(v, 1.5), lambda v: add_grain(v, 0.02, seed=1)],
    ids=["clean", "blur", "grain"],
)
def test_no_false_alarm_without_keyframes(clean, transform):
    assert find_keyframe_pulsing(transform(clean)) is None


def test_weak_pattern_is_below_the_screening_threshold():
    video = stepped_video(spike=0.011)
    assert find_keyframe_pulsing(video) is None
    assert find_keyframe_pulsing(video, min_strength=0.05).period == 12


def test_periods_without_enough_cycles_are_skipped():
    video = stepped_video(n_frames=31, period=12)
    assert find_keyframe_pulsing(video, min_cycles=4) is None
    assert find_keyframe_pulsing(video, min_cycles=2).period == 12


def test_static_video_has_no_pulsing():
    assert find_keyframe_pulsing(np.full((48, 8, 8), 0.5)) is None


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        frame_changes(np.zeros((1, 8, 8)))
    with pytest.raises(ValueError):
        find_keyframe_pulsing(np.zeros((48, 8, 8)), min_period=1)
    with pytest.raises(ValueError):
        find_keyframe_pulsing(np.zeros((48, 8, 8)), min_cycles=0)


def test_flicker_on_a_blurred_clip_can_trigger_a_false_alarm():
    # Known limitation, see find_keyframe_pulsing: any repeating change in the frame
    # change signal can pass the screen, not only keyframes.
    video = add_flicker(add_blur(textured_video(), 0.8), 0.03, 10.0, 24.0)
    assert find_keyframe_pulsing(video).period == 6


def test_short_periods_are_not_reported_by_default():
    video = stepped_video(period=3)
    assert find_keyframe_pulsing(video, min_period=2).period == 3
    assert find_keyframe_pulsing(video).period != 3
