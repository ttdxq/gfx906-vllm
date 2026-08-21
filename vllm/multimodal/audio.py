# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pybase64
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.mem_constants import MiB_bytes

from .base import MediaIO

logger = init_logger(__name__)

try:
    import av
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]


# Public libsndfile error codes that indicate client-provided format errors.
_BAD_SF_CODES = {0, 1, 3, 4}


def resample_audio_librosa(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
):
    # lazy import scipy.signal, otherwise it will crash doc build.
    import scipy.signal

    if orig_sr > target_sr:
        return scipy.signal.resample_poly(audio, 1, orig_sr // target_sr)
    elif orig_sr < target_sr:
        return scipy.signal.resample_poly(audio, target_sr // orig_sr, 1)
    return audio


class AudioResampler:
    """Resample audio data to a target sample rate."""

    def __init__(
        self,
        target_sr: float | None = None,
        method: Literal["librosa", "scipy"] = "librosa",
    ):
        self.target_sr = target_sr
        self.method = method

    def resample(
        self,
        audio: npt.NDArray[np.floating],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]:
        if self.target_sr is None:
            raise RuntimeError(
                "Audio resampling is not supported when `target_sr` is not provided"
            )
        if self.method == "librosa":
            return resample_audio_librosa(
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        elif self.method == "scipy":
            return resample_audio_scipy(
                audio, orig_sr=orig_sr, target_sr=self.target_sr
            )
        else:
            raise ValueError(
                f"Invalid resampling method: {self.method}. "
                "Supported methods are 'librosa' and 'scipy'."
            )


def load_audio_pyav(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> tuple[npt.NDArray, float]:
    """Load audio with PyAV while bounding decoded PCM allocation."""
    native_sr = None
    try:
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream found.")
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            native_sr = stream.rate
            sr = sr or native_sr

            if max_duration_s is not None and max_duration_s > 0:
                metadata_duration_s = None
                if stream.duration and stream.time_base:
                    metadata_duration_s = float(stream.duration * stream.time_base)
                elif container.duration:
                    metadata_duration_s = container.duration / 1_000_000
                if (
                    metadata_duration_s is not None
                    and metadata_duration_s > max_duration_s
                ):
                    raise ValueError(
                        f"Audio exceeds maximum allowed duration of "
                        f"{max_duration_s}s (metadata reports "
                        f"{metadata_duration_s:.1f}s). Set "
                        f"VLLM_MAX_AUDIO_DECODE_DURATION_S to increase this limit."
                    )

            max_samples = (
                int(sr * max_duration_s)
                if max_duration_s is not None and max_duration_s > 0
                else None
            )
            total_samples = 0
            total_decode_bytes = 0
            chunks: list[npt.NDArray] = []

            needs_resampling = not math.isclose(
                float(sr), float(native_sr), rel_tol=0.0, abs_tol=1e-6
            )
            resampler = (
                av.AudioResampler(format="fltp", layout="mono", rate=sr)
                if needs_resampling
                else None
            )
            for frame in container.decode(stream):
                if needs_resampling:
                    assert resampler is not None
                    arrays = (
                        out_frame.to_ndarray()
                        for out_frame in resampler.resample(frame)
                    )
                else:
                    arrays = (frame.to_ndarray(),)

                for array in arrays:
                    array = array.astype(np.float32, copy=False)
                    total_samples += array.shape[-1]
                    total_decode_bytes += array.nbytes
                    chunks.append(array)

                if max_samples is not None and total_samples > max_samples:
                    raise ValueError(
                        f"Audio exceeds maximum allowed duration of "
                        f"{max_duration_s}s (decoded {total_samples} samples at "
                        f"{sr}Hz). Set VLLM_MAX_AUDIO_DECODE_DURATION_S to "
                        f"increase this limit."
                    )
                if (
                    max_decode_bytes is not None
                    and max_decode_bytes > 0
                    and total_decode_bytes > max_decode_bytes
                ):
                    raise ValueError(
                        f"Audio decode exceeded "
                        f"{max_decode_bytes / MiB_bytes:.0f} MiB memory limit "
                        f"({total_decode_bytes / MiB_bytes:.0f} MiB decoded so "
                        f"far). Set VLLM_MAX_AUDIO_DECODE_BYTES to increase "
                        f"this limit."
                    )
    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise ValueError("Invalid or corrupted audio data.") from exc

    if not chunks:
        raise ValueError("No audio stream found.")

    audio = np.concatenate(chunks, axis=-1)
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    return audio, sr


def load_audio_soundfile(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio with SoundFile after validating its decoded size."""
    with soundfile.SoundFile(path) as audio_file:
        native_sr = audio_file.samplerate
        if max_duration_s is not None and max_duration_s > 0:
            file_duration_s = audio_file.frames / native_sr
            if file_duration_s > max_duration_s:
                raise ValueError(
                    f"Audio exceeds maximum allowed duration of "
                    f"{max_duration_s}s (file contains {file_duration_s:.1f}s "
                    f"at {native_sr}Hz). Set "
                    f"VLLM_MAX_AUDIO_DECODE_DURATION_S to increase this limit."
                )
        if max_decode_bytes is not None and max_decode_bytes > 0:
            estimated_bytes = (
                audio_file.frames * audio_file.channels * np.dtype(np.float32).itemsize
            )
            if estimated_bytes > max_decode_bytes:
                raise ValueError(
                    f"Audio would allocate {estimated_bytes / MiB_bytes:.0f} "
                    f"MiB of PCM ({audio_file.frames} frames x "
                    f"{audio_file.channels} channels x 4B), exceeding the "
                    f"{max_decode_bytes / MiB_bytes:.0f} MiB limit. Set "
                    f"VLLM_MAX_AUDIO_DECODE_BYTES to increase this limit."
                )
        audio = audio_file.read(dtype="float32", always_2d=False).T

    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=tuple(range(audio.ndim - 1)))

    if sr is not None and sr != native_sr:
        audio = resample_audio_librosa(audio, orig_sr=native_sr, target_sr=sr)
        return audio, int(sr)
    return audio, native_sr


def load_audio(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
) -> tuple[npt.NDArray, float]:
    """Load audio with SoundFile, falling back to PyAV when needed."""
    try:
        return load_audio_soundfile(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )
    except ImportError as exc:
        logger.error("Failed to load audio via soundfile: %r", exc)
    except soundfile.LibsndfileError as exc:
        if exc.code not in _BAD_SF_CODES:
            raise

    if isinstance(path, BytesIO):
        path.seek(0)
    try:
        return load_audio_pyav(
            path,
            sr=sr,
            mono=mono,
            max_duration_s=max_duration_s,
            max_decode_bytes=max_decode_bytes,
        )
    except ImportError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Invalid or unsupported audio file.") from exc


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        return load_audio(
            BytesIO(data),
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
            max_decode_bytes=envs.VLLM_MAX_AUDIO_DECODE_BYTES,
        )

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return load_audio(
            filepath,
            sr=None,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
            max_decode_bytes=envs.VLLM_MAX_AUDIO_DECODE_BYTES,
        )

    def encode_base64(self, media: tuple[npt.NDArray, int]) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            soundfile.write(buffer, audio, sr, format="WAV")
            data = buffer.getvalue()

        return base64.b64encode(data).decode("utf-8")


class AudioEmbeddingMediaIO(MediaIO[torch.Tensor]):
    def __init__(self) -> None:
        super().__init__()

    def load_bytes(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        return torch.load(buffer, weights_only=True)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        return torch.load(filepath, weights_only=True)

    def encode_base64(self, media: torch.Tensor) -> str:
        buffer = BytesIO()
        torch.save(media, buffer)
        buffer.seek(0)
        binary_data = buffer.read()
        return pybase64.b64encode(binary_data).decode("utf-8")
