"""A passthrough stream that fails after its keepalive has committed a 200.

The Anthropic `/anthropic/*` passthrough relays upstream bytes. Once the keepalive
has written `HTTP/1.1 200`, an upstream 4xx/5xx can no longer be expressed as a
status, so those bytes reach the caller as a bare JSON chunk inside an event
stream. Anthropic clients pick events by name and read that as a normal end.
"""

import asyncio
import json
from typing import AsyncIterator, Dict

import pytest
from starlette.responses import StreamingResponse

from litellm.proxy.common_request_processing import _anthropic_late_failure_frames, open_sse_before_first_byte
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import _anthropic_late_response_frames

PING = ": ping\n\n"


def _upstream_error_stream(payload: Dict[str, object], status_code: int = 500) -> StreamingResponse:
    body = json.dumps(payload).encode("utf-8")

    async def _body() -> AsyncIterator[bytes]:
        yield body

    return StreamingResponse(_body(), status_code=status_code, media_type="application/json")


async def _drain(response: StreamingResponse) -> str:
    chunks = [chunk async for chunk in response.body_iterator]
    return b"".join(chunks).decode("utf-8")


@pytest.mark.asyncio
async def test_anthropic_frames_replace_a_late_error_body():
    produced = _upstream_error_stream({"error": {"message": "upstream said no", "type": "server_error"}})

    async def slow_upstream():
        await asyncio.sleep(0.05)
        return produced

    response = await open_sse_before_first_byte(
        slow_upstream(),
        ping_interval_seconds=0.02,
        late_failure_frames=_anthropic_late_failure_frames,
        late_response_frames=_anthropic_late_response_frames,
    )

    body = await _drain(response)
    assert PING in body
    assert body.count("event: error") == 1
    assert "upstream said no" in body
    assert "[DONE]" not in body


@pytest.mark.asyncio
async def test_routes_without_an_adapter_still_relay_the_error_body_unchanged():
    produced = _upstream_error_stream({"error": {"message": "relay me as is"}})

    async def slow_upstream():
        await asyncio.sleep(0.05)
        return produced

    response = await open_sse_before_first_byte(slow_upstream(), ping_interval_seconds=0.02)

    body = await _drain(response)
    assert "event: error" not in body
    assert '"message": "relay me as is"' in body


@pytest.mark.asyncio
async def test_anthropic_late_response_frames_use_the_upstream_message():
    produced = _upstream_error_stream(
        {"error": {"message": "upstream said no", "type": "server_error"}}, status_code=529
    )

    frames = await _anthropic_late_response_frames(produced)

    assert len(frames) == 1
    assert frames[0].startswith("event: error")
    assert "upstream said no" in frames[0]


@pytest.mark.asyncio
async def test_a_body_that_is_not_json_still_produces_an_error_frame():
    async def _body() -> AsyncIterator[bytes]:
        yield b"<html>gateway timeout</html>"

    produced = StreamingResponse(_body(), status_code=504, media_type="text/html")

    frames = await _anthropic_late_response_frames(produced)

    assert len(frames) == 1
    assert "event: error" in frames[0]


@pytest.mark.asyncio
async def test_anthropic_late_failure_frames_cover_the_exception_path():
    async def boom():
        await asyncio.sleep(0.05)
        raise RuntimeError("upstream died after accepting the request")

    response = await open_sse_before_first_byte(
        boom(),
        ping_interval_seconds=0.02,
        late_failure_frames=_anthropic_late_failure_frames,
    )

    body = await _drain(response)
    assert body.count("event: error") == 1
    assert "[DONE]" not in body
