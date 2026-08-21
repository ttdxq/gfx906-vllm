# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

import vllm.envs as envs
from vllm.multimodal.image import ImageMediaIO, convert_image_mode

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()


def test_rgb_to_rgb():
    # Start with an RGB image.
    original_image = Image.open(ASSETS_DIR / "image1.png").convert("RGB")
    converted_image = convert_image_mode(original_image, "RGB")

    # RGB to RGB should be a no-op.
    diff = ImageChops.difference(original_image, converted_image)
    assert diff.getbbox() is None


def test_rgba_to_rgb():
    original_image = Image.open(ASSETS_DIR / "rgba.png")
    original_image_numpy = np.array(original_image)

    converted_image = convert_image_mode(original_image, "RGB")
    converted_image_numpy = np.array(converted_image)

    for i in range(original_image_numpy.shape[0]):
        for j in range(original_image_numpy.shape[1]):
            # Verify that all transparent pixels are converted to white.
            if original_image_numpy[i][j][3] == 0:
                assert converted_image_numpy[i][j][0] == 255
                assert converted_image_numpy[i][j][1] == 255
                assert converted_image_numpy[i][j][2] == 255


def test_rgba_to_rgb_custom_background(tmp_path):
    """Test RGBA to RGB conversion with custom background colors."""
    # Create a simple RGBA image with transparent and opaque pixels
    rgba_image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))  # Red with full opacity

    # Make top-left quadrant transparent
    for i in range(5):
        for j in range(5):
            rgba_image.putpixel((i, j), (0, 0, 0, 0))  # Fully transparent

    # Save the test image to tmp_path
    test_image_path = tmp_path / "test_rgba.png"
    rgba_image.save(test_image_path)

    # Test 1: Default white background (backward compatibility)
    image_io_default = ImageMediaIO()
    converted_default = image_io_default.load_file(test_image_path)
    default_numpy = np.array(converted_default)

    # Check transparent pixels are white
    assert default_numpy[0][0][0] == 255  # R
    assert default_numpy[0][0][1] == 255  # G
    assert default_numpy[0][0][2] == 255  # B
    # Check opaque pixels remain red
    assert default_numpy[5][5][0] == 255  # R
    assert default_numpy[5][5][1] == 0  # G
    assert default_numpy[5][5][2] == 0  # B

    # Test 2: Custom black background via kwargs
    image_io_black = ImageMediaIO(rgba_background_color=(0, 0, 0))
    converted_black = image_io_black.load_file(test_image_path)
    black_numpy = np.array(converted_black)

    # Check transparent pixels are black
    assert black_numpy[0][0][0] == 0  # R
    assert black_numpy[0][0][1] == 0  # G
    assert black_numpy[0][0][2] == 0  # B
    # Check opaque pixels remain red
    assert black_numpy[5][5][0] == 255  # R
    assert black_numpy[5][5][1] == 0  # G
    assert black_numpy[5][5][2] == 0  # B

    # Test 3: Custom blue background via kwargs (as list)
    image_io_blue = ImageMediaIO(rgba_background_color=[0, 0, 255])
    converted_blue = image_io_blue.load_file(test_image_path)
    blue_numpy = np.array(converted_blue)

    # Check transparent pixels are blue
    assert blue_numpy[0][0][0] == 0  # R
    assert blue_numpy[0][0][1] == 0  # G
    assert blue_numpy[0][0][2] == 255  # B

    # Test 4: Test with load_bytes method
    with open(test_image_path, "rb") as f:
        image_data = f.read()

    image_io_green = ImageMediaIO(rgba_background_color=(0, 255, 0))
    converted_green = image_io_green.load_bytes(image_data)
    green_numpy = np.array(converted_green)

    # Check transparent pixels are green
    assert green_numpy[0][0][0] == 0  # R
    assert green_numpy[0][0][1] == 255  # G
    assert green_numpy[0][0][2] == 0  # B


def test_rgba_background_color_validation():
    """Test that invalid rgba_background_color values are properly rejected."""

    # Test invalid types
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color="255,255,255")

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=255)

    # Test wrong number of elements
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255, 255, 255))

    # Test non-integer values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255.0, 255.0, 255.0))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, "255", 255))

    # Test out of range values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(256, 255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, -1, 255))

    # Test that valid values work
    ImageMediaIO(rgba_background_color=(0, 0, 0))  # Should not raise
    ImageMediaIO(rgba_background_color=[255, 255, 255])  # Should not raise
    ImageMediaIO(rgba_background_color=(128, 128, 128))  # Should not raise


def _encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_pixel_limit_respected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 10_000, raising=False)
    data = _encode_png(Image.new("RGB", (100, 100), (255, 0, 0)))

    result = ImageMediaIO().load_bytes(data)

    assert result.size == (100, 100)


def test_image_pixel_limit_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100, raising=False)
    data = _encode_png(Image.new("RGB", (20, 20), (0, 255, 0)))

    with pytest.raises(ValueError, match="exceed"):
        ImageMediaIO().load_bytes(data)


def test_image_file_pixel_limit_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100, raising=False)
    image_path = tmp_path / "oversized.png"
    Image.new("RGB", (20, 20), (0, 255, 0)).save(image_path)

    with pytest.raises(ValueError, match="exceed"):
        ImageMediaIO().load_file(image_path)


def test_image_pixel_limit_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 0, raising=False)
    data = _encode_png(Image.new("RGB", (1000, 1000), (0, 0, 255)))

    result = ImageMediaIO().load_bytes(data)

    assert result.size == (1000, 1000)
