# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

import vllm.entrypoints.openai.api_server as api_server
from vllm.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.speech_to_text import read_upload_with_limit
from vllm.exceptions import VLLMValidationError


def _make_upload(data: bytes, *, size: int | None) -> AsyncMock:
    upload = AsyncMock()
    upload.size = size
    offset = 0

    async def read(count: int = -1) -> bytes:
        nonlocal offset
        if count < 0:
            count = len(data) - offset
        chunk = data[offset : offset + count]
        offset += len(chunk)
        return chunk

    upload.read = AsyncMock(side_effect=read)
    return upload


@pytest.mark.asyncio
async def test_known_oversized_upload_rejected_without_reading():
    upload = _make_upload(b"", size=1025)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await read_upload_with_limit(upload, max_size_mb=1 / 1024)

    upload.read.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_oversized_upload_stops_early():
    data = b"x" * (256 * 1024)
    upload = _make_upload(data, size=None)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await read_upload_with_limit(upload, max_size_mb=128 / 1024)

    assert upload.read.call_count == 3


@pytest.mark.asyncio
async def test_upload_at_exact_limit_allowed():
    data = b"x" * (128 * 1024)
    upload = _make_upload(data, size=None)

    assert await read_upload_with_limit(upload, max_size_mb=128 / 1024) == data


def _original_endpoint(function):
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_name", "factory_name", "method_name"),
    [
        ("create_transcriptions", "transcription", "create_transcription"),
        ("create_translations", "translation", "create_translation"),
    ],
)
async def test_audio_routes_reject_oversized_upload_before_handler(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
    factory_name: str,
    method_name: str,
):
    operation = AsyncMock()
    handler = Mock(**{method_name: operation})
    monkeypatch.setattr(api_server, factory_name, lambda _: handler)
    request = SimpleNamespace(file=_make_upload(b"", size=26 * 1024**2))

    endpoint = _original_endpoint(getattr(api_server, endpoint_name))
    with pytest.raises(HTTPException) as exc_info:
        if endpoint_name == "create_translations":
            await endpoint(request, Mock())
        else:
            await endpoint(Mock(), request)

    assert exc_info.value.status_code == 400
    operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_decode_validation_is_bad_request(monkeypatch: pytest.MonkeyPatch):
    operation = AsyncMock(
        return_value=ErrorResponse(
            error=ErrorInfo(
                message="audio limit exceeded",
                type="BadRequestError",
                code=400,
            )
        )
    )
    handler = Mock(create_transcription=operation)
    monkeypatch.setattr(api_server, "transcription", lambda _: handler)
    request = SimpleNamespace(file=_make_upload(b"audio", size=5))

    endpoint = _original_endpoint(api_server.create_transcriptions)
    response = await endpoint(Mock(), request)

    assert response.status_code == 400
    operation.assert_awaited_once()
