"""Compare a test video with a reference using all metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .luma import as_luma, image_as_luma
from .pulsing import PulsingEstimate, find_keyframe_pulsing
from .spatial import band_energies, radial_power_spectrum, video_radial_spectrum
from .temporal import DEFAULT_CUTOFF_HZ, temporal_bands, temporal_spectrum


def _safe_ratio(test: float, reference: float) -> float:
    # A reference with no energy in a band, such as a flat image, has no meaningful ratio.
    return test / reference if reference > 0 else float("nan")


@dataclass(frozen=True)
class EvaluationReport:
    """Test-to-reference ratios, where 1.0 means the same energy as the reference."""

    spatial_ratio: dict[str, float]
    temporal_ratio: dict[str, float]
    pulsing_test: PulsingEstimate | None
    pulsing_reference: PulsingEstimate | None
    cutoff_hz: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines = ["Spatial energy, test / reference"]
        lines += [f"  {name:<5} {value:.3f}" for name, value in self.spatial_ratio.items()]
        lines.append(f"Temporal energy, test / reference (cutoff {self.cutoff_hz:g} Hz)")
        lines += [f"  {name:<5} {value:.3f}" for name, value in self.temporal_ratio.items()]

        def pulsing(estimate: PulsingEstimate | None) -> str:
            if estimate is None:
                return "none found"
            return f"period {estimate.period}, strength {estimate.strength:.2f}"

        lines.append("Keyframe pulsing")
        lines.append(f"  test       {pulsing(self.pulsing_test)}")
        lines.append(f"  reference  {pulsing(self.pulsing_reference)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ImageReport:
    """Test-to-reference energy per spatial band, where 1.0 means the same as the reference."""

    spatial_ratio: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines = ["Spatial energy, test / reference"]
        lines += [f"  {name:<5} {value:.3f}" for name, value in self.spatial_ratio.items()]
        return "\n".join(lines)


def compare_images(test: np.ndarray, reference: np.ndarray) -> ImageReport:
    """Compare the spatial bands of two images of the same size.

    Flicker needs more than one frame, so it is not available here.
    """
    # Limitations:
    # - Ratios describe a difference from the reference, so one is required, with the same size.
    # - Tested on degradations applied to real photos, not on real model output.
    # - See band_energies for the limits of the bands.
    test_luma, reference_luma = image_as_luma(test), image_as_luma(reference)
    if test_luma.shape != reference_luma.shape:
        raise ValueError(
            f"image sizes differ: {test_luma.shape} and {reference_luma.shape}. "
            "Spatial frequencies are per pixel, so resize the images first."
        )
    test_bands = band_energies(radial_power_spectrum(test_luma))
    reference_bands = band_energies(radial_power_spectrum(reference_luma))
    return ImageReport(
        spatial_ratio={k: _safe_ratio(test_bands[k], reference_bands[k]) for k in test_bands}
    )


def compare_videos(
    test: np.ndarray,
    reference: np.ndarray,
    fps: float,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    max_frames: int | None = None,
) -> EvaluationReport:
    """Compute every metric for a test video against a reference of the same size."""
    # Limitations:
    # - Ratios describe a difference from the reference, so one is required, with the same frame
    #   size. Pulsing itself needs no reference, but comparing shows who introduced the pattern.
    # - Video metrics were tested on synthetic clips, not on real generated video. They show that
    #   each metric responds to its target defect, not how well it tracks human judgement.
    # - See band_energies, temporal_bands and find_keyframe_pulsing for each metric's limits.
    test_luma, reference_luma = as_luma(test), as_luma(reference)
    if test_luma.shape[1:] != reference_luma.shape[1:]:
        raise ValueError(
            f"frame sizes differ: {test_luma.shape[1:]} and {reference_luma.shape[1:]}. "
            "Spatial frequencies are per pixel, so resize the videos first."
        )

    test_bands = band_energies(video_radial_spectrum(test_luma, max_frames))
    reference_bands = band_energies(video_radial_spectrum(reference_luma, max_frames))
    test_temporal = temporal_bands(temporal_spectrum(test_luma, fps), cutoff_hz)
    reference_temporal = temporal_bands(temporal_spectrum(reference_luma, fps), cutoff_hz)

    return EvaluationReport(
        spatial_ratio={k: _safe_ratio(test_bands[k], reference_bands[k]) for k in test_bands},
        temporal_ratio={k: _safe_ratio(test_temporal[k], reference_temporal[k]) for k in test_temporal},
        pulsing_test=find_keyframe_pulsing(test_luma),
        pulsing_reference=find_keyframe_pulsing(reference_luma),
        cutoff_hz=cutoff_hz,
    )
