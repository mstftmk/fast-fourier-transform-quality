import json

import numpy as np
import pytest

from fast_fourier_transform_quality.cli import main
from fast_fourier_transform_quality.io import load_video
from fast_fourier_transform_quality.synthetic import add_blur, textured_video


@pytest.fixture
def npy_pair(tmp_path):
    clean = textured_video(n_frames=48, height=48, width=48)
    test_path, reference_path = tmp_path / "test.npy", tmp_path / "reference.npy"
    np.save(test_path, add_blur(clean, 1.5))
    np.save(reference_path, clean)
    return str(test_path), str(reference_path)


def test_compare_prints_json(npy_pair, capsys):
    assert main(["compare", *npy_pair, "--fps", "24", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["spatial_ratio"]["high"] < 0.1


def test_compare_prints_text_and_honours_max_frames(npy_pair, capsys):
    assert main(["compare", *npy_pair, "--fps", "24", "--max-frames", "16"]) == 0
    assert "Spatial energy" in capsys.readouterr().out


def test_npy_without_fps_fails_with_a_clear_message(npy_pair, capsys):
    assert main(["compare", *npy_pair]) == 2
    assert "--fps" in capsys.readouterr().err


def test_missing_file_fails_cleanly(tmp_path, capsys):
    missing = str(tmp_path / "missing.npy")
    assert main(["compare", missing, missing, "--fps", "24"]) == 2
    assert "error:" in capsys.readouterr().err


def test_validate_prints_the_table(capsys):
    assert main(["validate"]) == 0
    assert "Flicker (10 Hz)" in capsys.readouterr().out


def test_video_file_round_trip(tmp_path, capsys):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    frames = (np.clip(textured_video(n_frames=24, height=64, width=64), 0, 1) * 255).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("MJPG writer not available")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    loaded, fps = load_video(path, max_frames=12)
    assert loaded.shape == (12, 64, 64, 3)
    assert fps == pytest.approx(24.0)
    assert main(["compare", str(path), str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["spatial_ratio"]["high"] == pytest.approx(1.0)
