"""
Eval LR-14: Streaming API contract.

Verify that POST /api/v1/chat returns SSE-formatted chunks with
the correct content type and a [DONE] marker.

This is an API contract test, not a conversation eval.

Safety (100% required):
  - Content-Type is text/event-stream
  - Response contains data: chunks
  - Response ends with data: [DONE]
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EVALS") != "1",
    reason="Eval suite is opt-in. Set RUN_EVALS=1 to run.",
)

BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000")


@pytest.mark.asyncio
async def test_streaming_sse_contract():
    """POST /api/v1/chat should return SSE with data: chunks and [DONE]."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        resp = await client.post(
            "/api/v1/chat",
            json={"message": "hello", "thread_id": "eval-streaming-test"},
        )

        # Check content type
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"Expected text/event-stream, got: {content_type}"
        )

        body = resp.text

        # Check for data: chunks
        data_lines = [line for line in body.splitlines() if line.startswith("data:")]
        assert len(data_lines) > 0, "Expected at least one data: chunk in SSE response"

        # Check for [DONE] marker
        assert any("[DONE]" in line for line in data_lines), (
            "Expected data: [DONE] marker at end of SSE stream"
        )
