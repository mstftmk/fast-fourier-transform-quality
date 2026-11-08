"""Encode panning clips with x264 at known keyframe intervals and test find_keyframe_pulsing.

Regenerates docs/keyframe_example.png and prints a table. Needs ffmpeg with libx264 and
pip install -e ".[examples]". The photos are scikit-image's "coffee" (CC0, Rachel Michetti,
courtesy of Pikolo Espresso Bar) and "chelsea" (CC0, Stefan van der Walt). Exact results
can change with the x264 version.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from fast_fourier_transform_quality import find_keyframe_pulsing, frame_changes
from fast_fourier_transform_quality.pulsing import autocorrelation

SIZE = 256
N_FRAMES = 360
INTERVALS = (8, 12, 16, 24, 48)
NO_KEYFRAMES = 100_000


def panning_clip(photo: np.ndarray) -> np.ndarray:
    """Pan over a center crop with a Fourier shift, so no interpolation pattern is added."""
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(float)
    top, left = (gray.shape[0] - SIZE) // 2, (gray.shape[1] - SIZE) // 2
    spectrum = np.fft.fft2(gray[top : top + SIZE, left : left + SIZE])
    fy, fx = np.meshgrid(np.fft.fftfreq(SIZE), np.fft.fftfreq(SIZE), indexing="ij")
    frames = np.empty((N_FRAMES, SIZE, SIZE), np.uint8)
    for t in range(N_FRAMES):
        shifted = np.fft.ifft2(spectrum * np.exp(-2j * np.pi * t * (fy * 0.1913 + fx * 0.3731)))
        frames[t] = np.clip(shifted.real + 0.5, 0, 255)
    return frames


def encode(frames: np.ndarray, interval: int, crf: int, b_frames: int) -> np.ndarray:
    """Encode with x264 at a fixed keyframe interval and decode back to grayscale frames."""
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "clip.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray"]
            + ["-s", f"{SIZE}x{SIZE}", "-r", "24", "-i", "-", "-c:v", "libx264"]
            + ["-preset", "medium", "-pix_fmt", "yuv420p", "-crf", str(crf), "-bf", str(b_frames)]
            + ["-g", str(interval), "-keyint_min", str(interval), "-sc_threshold", "0", path],
            input=frames.tobytes(),
            check=True,
        )
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True,
            check=True,
        ).stdout
    return np.frombuffer(decoded, np.uint8).reshape(-1, SIZE, SIZE)


def main() -> None:
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    print(version.stdout.splitlines()[0].split(" Copyright")[0], "\n")
    header = " | ".join(str(i) for i in INTERVALS)
    print(f"| Photo | CRF | B-frames | {header} | No keyframes |")
    print("|---|---:|---:|" + "---:|" * len(INTERVALS) + "---:|")

    clips = {"coffee": panning_clip(data.coffee()), "chelsea": panning_clip(data.chelsea())}
    correct, total, alarms = 0, 0, 0
    for name, clip in clips.items():
        for crf in (38, 28):
            for b_frames in (0, 3):
                cells = []
                for interval in INTERVALS:
                    estimate = find_keyframe_pulsing(encode(clip, interval, crf, b_frames))
                    found = None if estimate is None else estimate.period
                    correct += found == interval
                    total += 1
                    cells.append("none" if found is None else str(found))
                control = find_keyframe_pulsing(encode(clip, NO_KEYFRAMES, crf, b_frames))
                alarms += control is not None
                cells.append("none" if control is None else f"period {control.period}")
                print(f"| {name} | {crf} | {b_frames} | " + " | ".join(cells) + " |")
    print(f"\nexact period: {correct} of {total}; alarms without keyframes: {alarms} of 8")
    for name, clip in clips.items():
        print(f"source {name}: {find_keyframe_pulsing(clip)}")

    keyed = frame_changes(encode(clips["coffee"], 24, 38, 0))
    plain = frame_changes(encode(clips["coffee"], NO_KEYFRAMES, 38, 0))
    fig, (changes, correlation) = plt.subplots(1, 2, figsize=(10, 3.6))
    changes.plot(keyed[:121], label="Keyframe every 24 frames")
    changes.plot(plain[:121], label="No keyframes")
    changes.set(title="Keyframes make the change spike", xlabel="frame step", ylabel="RMS change")
    correlation.plot(autocorrelation(keyed)[:90], label="Keyframe every 24 frames")
    correlation.plot(autocorrelation(plain)[:90], label="No keyframes")
    correlation.set(title="Autocorrelation peaks at the interval", xlabel="lag (frames)")
    correlation.set_ylim(-1, 1)
    for axis in (changes, correlation):
        axis.legend(frameon=False, loc="upper right")
        axis.grid(alpha=0.3)
    changes.set_ylim(top=max(keyed[:121]) * 1.25)
    fig.tight_layout()
    output = Path(__file__).parents[1] / "docs" / "keyframe_example.png"
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
