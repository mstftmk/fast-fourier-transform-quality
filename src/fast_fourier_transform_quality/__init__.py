"""Fast Fourier Transform metrics for judging images and video against a reference."""

from .pulsing import PulsingEstimate, find_keyframe_pulsing, frame_changes
from .report import EvaluationReport, ImageReport, compare_images, compare_videos
from .spatial import band_energies, radial_power_spectrum, video_radial_spectrum
from .temporal import temporal_bands, temporal_spectrum

__all__ = [
    "EvaluationReport",
    "ImageReport",
    "PulsingEstimate",
    "band_energies",
    "compare_images",
    "compare_videos",
    "find_keyframe_pulsing",
    "frame_changes",
    "radial_power_spectrum",
    "temporal_bands",
    "temporal_spectrum",
    "video_radial_spectrum",
]

__version__ = "0.1.0"
