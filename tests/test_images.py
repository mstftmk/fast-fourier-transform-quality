import json

import numpy as np
import pytest

from fast_fourier_transform_quality import compare_images
from fast_fourier_transform_quality.cli import main
from fast_fourier_transform_quality.luma import image_as_luma
from fast_fourier_transform_quality.synthetic import add_blur, textured_video


@pytest.fixture(scope="module")
def image():
    return textured_video(n_frames=1)[0]


def test_identical_images_score_one(image):
    report = compare_images(image, image)
    assert all(v == pytest.approx(1.0) for v in report.spatial_ratio.values())


def test_blur_lowers_fine_bands_most(image):
    blurred = add_blur(image[None], 1.5)[0]
    ratios = compare_images(blurred, image).spatial_ratio
    assert ratios["high"] < ratios["mid"] < ratios["low"] < 1.0


def test_noise_raises_the_high_band(image):
    noisy = image + 0.03 * np.random.default_rng(0).standard_normal(image.shape)
    assert compare_images(noisy, image).spatial_ratio["high"] > 1.5


def test_rgb_and_grayscale_inputs_agree(image):
    rgb = np.repeat(image[..., None], 3, axis=-1)
    gray_ratio = compare_images(add_blur(image[None], 1.0)[0], image).spatial_ratio["high"]
    rgb_ratio = compare_images(np.repeat(add_blur(image[None], 1.0)[0][..., None], 3, axis=-1), rgb)
    assert rgb_ratio.spatial_ratio["high"] == pytest.approx(gray_ratio)


def test_non_square_images_are_supported(image):
    crop = image[:64, :128]
    assert compare_images(crop, crop).spatial_ratio["high"] == pytest.approx(1.0)


def test_different_sizes_are_rejected(image):
    with pytest.raises(ValueError, match="image sizes differ"):
        compare_images(image, image[:64, :64])


@pytest.mark.parametrize("shape", [(32,), (32, 32, 4), (2, 32, 32, 3)])
def test_rejects_wrong_shapes(shape):
    with pytest.raises(ValueError):
        image_as_luma(np.zeros(shape))


def test_report_serializes(image):
    report = compare_images(image, image)
    assert json.loads(json.dumps(report.to_dict()))["spatial_ratio"]["high"] == pytest.approx(1.0)
    assert "Spatial energy" in report.to_text()


def test_cli_compares_npy_images(image, tmp_path, capsys):
    test_path, reference_path = tmp_path / "test.npy", tmp_path / "reference.npy"
    np.save(test_path, add_blur(image[None], 1.5)[0])
    np.save(reference_path, image)
    assert main(["compare-images", str(test_path), str(reference_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["spatial_ratio"]["high"] < 0.1


def test_cli_compares_png_files(image, tmp_path, capsys):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), (np.clip(image, 0, 1) * 255).astype(np.uint8))
    assert main(["compare-images", str(path), str(path)]) == 0
    assert "high" in capsys.readouterr().out


def test_cli_reports_unreadable_image(tmp_path, capsys):
    pytest.importorskip("cv2")
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert main(["compare-images", str(path), str(path)]) == 2
    assert "could not read image" in capsys.readouterr().err
