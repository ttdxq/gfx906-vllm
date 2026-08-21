# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pybase64
import pytest
from PIL import Image

import vllm.envs as envs
from vllm.assets.base import get_vllm_public_assets
from vllm.assets.video import video_to_ndarrays, video_to_pil_images_list
from vllm.multimodal.image import ImageMediaIO
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    OpenCVDynamicVideoBackend,
    OpenCVVideoBackend,
    VideoLoader,
    VideoMediaIO,
)

from .utils import cosine_similarity, create_video_from_image, normalize_image

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
NUM_FRAMES = 10
FAKE_OUTPUT_1 = np.random.rand(NUM_FRAMES, 1280, 720, 3)
FAKE_OUTPUT_2 = np.random.rand(NUM_FRAMES, 1280, 720, 3)


@VIDEO_LOADER_REGISTRY.register("test_video_loader_1")
class TestVideoLoader1(VideoLoader):
    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
        return FAKE_OUTPUT_1


@VIDEO_LOADER_REGISTRY.register("test_video_loader_2")
class TestVideoLoader2(VideoLoader):
    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1) -> npt.NDArray:
        return FAKE_OUTPUT_2


def test_video_loader_registry():
    custom_loader_1 = VIDEO_LOADER_REGISTRY.load("test_video_loader_1")
    output_1 = custom_loader_1.load_bytes(b"test")
    np.testing.assert_array_equal(output_1, FAKE_OUTPUT_1)

    custom_loader_2 = VIDEO_LOADER_REGISTRY.load("test_video_loader_2")
    output_2 = custom_loader_2.load_bytes(b"test")
    np.testing.assert_array_equal(output_2, FAKE_OUTPUT_2)


def test_video_loader_type_doesnt_exist():
    with pytest.raises(AssertionError):
        VIDEO_LOADER_REGISTRY.load("non_existing_video_loader")


@VIDEO_LOADER_REGISTRY.register("assert_10_frames_1_fps")
class Assert10Frames1FPSVideoLoader(VideoLoader):
    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, fps: float = -1.0, **kwargs
    ) -> npt.NDArray:
        assert num_frames == 10, "bad num_frames"
        assert fps == 1.0, "bad fps"
        return FAKE_OUTPUT_2


def test_video_media_io_kwargs(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "assert_10_frames_1_fps")
        imageio = ImageMediaIO()

        # Verify that different args pass/fail assertions as expected.
        videoio = VideoMediaIO(imageio, **{"num_frames": 10, "fps": 1.0})
        _ = videoio.load_bytes(b"test")

        videoio = VideoMediaIO(
            imageio, **{"num_frames": 10, "fps": 1.0, "not_used": "not_used"}
        )
        _ = videoio.load_bytes(b"test")

        with pytest.raises(AssertionError, match="bad num_frames"):
            videoio = VideoMediaIO(imageio, **{})
            _ = videoio.load_bytes(b"test")

        with pytest.raises(AssertionError, match="bad num_frames"):
            videoio = VideoMediaIO(imageio, **{"num_frames": 9, "fps": 1.0})
            _ = videoio.load_bytes(b"test")

        with pytest.raises(AssertionError, match="bad fps"):
            videoio = VideoMediaIO(imageio, **{"num_frames": 10, "fps": 2.0})
            _ = videoio.load_bytes(b"test")


@pytest.mark.parametrize("is_color", [True, False])
@pytest.mark.parametrize("fourcc, ext", [("mp4v", "mp4"), ("XVID", "avi")])
def test_opencv_video_io_colorspace(is_color: bool, fourcc: str, ext: str):
    """
    Test all functions that use OpenCV for video I/O return RGB format.
    Both RGB and grayscale videos are tested.
    """
    image_path = get_vllm_public_assets(
        filename="stop_sign.jpg", s3_prefix="vision_model_images"
    )
    image = Image.open(image_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        if not is_color:
            image_path = f"{tmpdir}/test_grayscale_image.png"
            image = image.convert("L")
            image.save(image_path)
            # Convert to gray RGB for comparison
            image = image.convert("RGB")
        video_path = f"{tmpdir}/test_RGB_video.{ext}"
        create_video_from_image(
            image_path,
            video_path,
            num_frames=2,
            is_color=is_color,
            fourcc=fourcc,
        )

        frames = video_to_ndarrays(video_path)
        for frame in frames:
            sim = cosine_similarity(
                normalize_image(np.array(frame)), normalize_image(np.array(image))
            )
            assert np.sum(np.isnan(sim)) / sim.size < 0.001
            assert np.nanmean(sim) > 0.99

        pil_frames = video_to_pil_images_list(video_path)
        for frame in pil_frames:
            sim = cosine_similarity(
                normalize_image(np.array(frame)), normalize_image(np.array(image))
            )
            assert np.sum(np.isnan(sim)) / sim.size < 0.001
            assert np.nanmean(sim) > 0.99

        io_frames, _ = VideoMediaIO(ImageMediaIO()).load_file(Path(video_path))
        for frame in io_frames:
            sim = cosine_similarity(
                normalize_image(np.array(frame)), normalize_image(np.array(image))
            )
            assert np.sum(np.isnan(sim)) / sim.size < 0.001
            assert np.nanmean(sim) > 0.99


def test_video_backend_handles_broken_frames(monkeypatch: pytest.MonkeyPatch):
    """
    Regression test for handling videos with broken frames.
    This test uses a pre-corrupted video file (assets/corrupted.mp4) that
    contains broken/unreadable frames to verify the video loader handles
    them gracefully without crashing and returns accurate metadata.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")

        # Load the pre-corrupted video file that contains broken frames
        corrupted_video_path = ASSETS_DIR / "corrupted.mp4"

        with open(corrupted_video_path, "rb") as f:
            video_data = f.read()

        loader = VIDEO_LOADER_REGISTRY.load("opencv")
        frames, metadata = loader.load_bytes(video_data, num_frames=-1)

        # Verify metadata consistency:
        # frames_indices must match actual loaded frames
        assert frames.shape[0] == len(metadata["frames_indices"]), (
            f"Frames array size must equal frames_indices length. "
            f"Got {frames.shape[0]} frames but "
            f"{len(metadata['frames_indices'])} indices"
        )

        # Verify that broken frames were skipped:
        # loaded frames should be less than total
        assert frames.shape[0] < metadata["total_num_frames"], (
            f"Should load fewer frames than total due to broken frames. "
            f"Expected fewer than {metadata['total_num_frames']} frames, "
            f"but loaded {frames.shape[0]} frames"
        )


def _make_jpeg_b64_frames(n: int, width: int = 8, height: int = 8) -> list[str]:
    frames = []
    for i in range(n):
        image = Image.new("RGB", (width, height), color=(i % 256, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        frames.append(pybase64.b64encode(buffer.getvalue()).decode("ascii"))
    return frames


def test_load_base64_jpeg_enforces_num_frames_limit():
    data = ",".join(_make_jpeg_b64_frames(20))

    frames, metadata = VideoMediaIO(ImageMediaIO(), num_frames=4).load_base64(
        "video/jpeg", data
    )

    assert frames.shape[0] == 4
    assert metadata == {}


def test_load_base64_jpeg_no_limit_when_num_frames_negative():
    data = ",".join(_make_jpeg_b64_frames(10))

    frames, metadata = VideoMediaIO(ImageMediaIO(), num_frames=-1).load_base64(
        "video/jpeg", data
    )

    assert frames.shape[0] == 10
    assert metadata == {}


def test_load_base64_jpeg_raises_on_zero_num_frames():
    data = ",".join(_make_jpeg_b64_frames(3))

    with pytest.raises(ValueError, match="num_frames must be greater than 0 or -1"):
        VideoMediaIO(ImageMediaIO(), num_frames=0).load_base64("video/jpeg", data)


@pytest.mark.parametrize("backend_cls", [OpenCVVideoBackend, OpenCVDynamicVideoBackend])
def test_opencv_backends_reject_oversized_frames(
    monkeypatch: pytest.MonkeyPatch, backend_cls: type[VideoLoader]
):
    import cv2

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 20
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 20
            raise AssertionError("frame limit must be checked before other metadata")

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100, raising=False)
    monkeypatch.setattr(backend_cls, "get_cv2_video_api", lambda self: None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: FakeCapture())

    with pytest.raises(ValueError, match="Video frame dimensions 20x20"):
        backend_cls.load_bytes(b"video")
