"""The frame a late stream failure travels in must match the convention of whoever is reading.

An Anthropic client raises only on `event: error`; a bare `data: {"error": ...}` frame is an unknown
event and is skipped, so a failure delivered that way never reaches the caller.
"""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from litellm.proxy.common_request_processing import (
    _anthropic_sse_error_frames,
    _late_failure_frames,
    _streams_anthropic_events,
)


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def test_anthropic_paths_are_recognised() -> None:
    assert _streams_anthropic_events(_request("/v1/messages"))
    assert _streams_anthropic_events(_request("/v1/messages/"))
    assert _streams_anthropic_events(_request("/anthropic/v1/messages"))


def test_openai_paths_are_left_alone() -> None:
    assert not _streams_anthropic_events(_request("/v1/chat/completions"))
    assert not _streams_anthropic_events(None)


def test_a_request_without_a_real_path_is_not_anthropic() -> None:
    """A mocked or partial request must not be read as an Anthropic caller."""

    class _MockedRequest:
        url = MagicMock()

    assert not _streams_anthropic_events(cast(Request, _MockedRequest()))
    assert not _streams_anthropic_events(cast(Request, MagicMock()))


def test_anthropic_frame_is_an_error_event_without_done() -> None:
    frames = _anthropic_sse_error_frames({"message": "boom", "type": "internal_server_error", "code": "500"})

    assert len(frames) == 1
    assert frames[0].startswith("event: error\n")
    assert "[DONE]" not in frames[0]
    payload = json.loads(frames[0].split("data: ", 1)[1])
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "api_error"
    assert payload["error"]["message"] == "boom"


def test_status_maps_to_the_anthropic_error_type() -> None:
    payload = json.loads(_anthropic_sse_error_frames({"code": "429"})[0].split("data: ", 1)[1])

    assert payload["error"]["type"] == "rate_limit_error"


@pytest.mark.parametrize(
    ("path", "expect_anthropic"),
    [("/v1/messages", True), ("/v1/chat/completions", False)],
)
def test_the_caller_decides_the_convention(path: str, expect_anthropic: bool) -> None:
    frames = _late_failure_frames({"message": "boom", "code": "500"}, _request(path))

    assert frames[0].startswith("event: error") is expect_anthropic
    assert ("[DONE]" in "".join(frames)) is not expect_anthropic


async def test_a_failure_after_the_first_ping_reaches_an_anthropic_caller() -> None:
    """The measured case: keepalives already on the wire, upstream fails, caller must still see it."""
    import asyncio

    from litellm.proxy.common_request_processing import open_sse_before_first_byte

    async def failing() -> object:
        await asyncio.sleep(5)
        raise RuntimeError("upstream died")

    response = await open_sse_before_first_byte(
        failing(),
        ping_interval_seconds=0.01,
        request=_request("/v1/messages"),
    )

    body = b""
    async for chunk in response.body_iterator:
        body += chunk.encode() if isinstance(chunk, str) else chunk

    assert b": ping" in body
    assert b"event: error" in body
    assert b"[DONE]" not in body
