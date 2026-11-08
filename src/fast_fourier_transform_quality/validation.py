"""Apply each known defect to a synthetic clip and tabulate how every metric responds."""

from __future__ import annotations

from .report import EvaluationReport, compare_videos
from .synthetic import add_blur, add_flicker, add_grain, add_keyframe_pulsing, textured_video

FPS = 24.0


def run_controls(seed: int = 0) -> dict[str, EvaluationReport]:
    """Return a report for each control, always compared with the same clean clip."""
    clean = textured_video(seed=seed)
    tests = {
        "No change": clean,
        "Blur (sigma 1.5 px)": add_blur(clean, 1.5),
        "Flicker (10 Hz)": add_flicker(clean, 0.03, 10.0, FPS),
        "Keyframe pulsing (12 frames)": add_keyframe_pulsing(clean, 12, 1.0),
        "Grain": add_grain(clean, 0.02, seed=seed + 1),
        "Different content": textured_video(seed=seed + 100),
    }
    return {name: compare_videos(test, clean, FPS) for name, test in tests.items()}


def controls_table(seed: int = 0) -> str:
    """Format the controls as a Markdown table."""
    rows = [
        "| Control | Low | Mid | High | Slow | Fast | Pulsing |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, report in run_controls(seed).items():
        spatial, temporal = report.spatial_ratio, report.temporal_ratio
        estimate = report.pulsing_test
        found = "none" if estimate is None else f"period {estimate.period}"
        rows.append(
            f"| {name} | {spatial['low']:.2f} | {spatial['mid']:.2f} | {spatial['high']:.2f} "
            f"| {temporal['slow']:.2f} | {temporal['fast']:.2f} | {found} |"
        )
    return "\n".join(rows)
