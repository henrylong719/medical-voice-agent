"""
Eval LR-09: Admin DOB validation.

The admin patient search route should reject non-ISO date formats
with 422, while accepting valid ISO YYYY-MM-DD format.

This is a simple API contract test, not a conversation eval.

Safety (100% required):
  - Non-ISO DOB returns 422
  - ISO DOB returns 200
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
async def test_admin_rejects_non_iso_dob():
    """POST /api/v1/admin/patients/search with slash-formatted DOB should 422."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Non-ISO format should be rejected
        resp = await client.post(
            "/api/v1/admin/patients/search",
            json={
                "full_name": "Alice Johnson",
                "date_of_birth": "05/14/1992",
            },
        )
        assert resp.status_code == 422, (
            f"Expected 422 for non-ISO DOB, got {resp.status_code}: {resp.text}"
        )


@pytest.mark.asyncio
async def test_admin_accepts_iso_dob():
    """POST /api/v1/admin/patients/search with ISO DOB should succeed."""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        resp = await client.post(
            "/api/v1/admin/patients/search",
            json={
                "full_name": "Alice Johnson",
                "date_of_birth": "1992-05-14",
            },
        )
        assert resp.status_code == 200, (
            f"Expected 200 for ISO DOB, got {resp.status_code}: {resp.text}"
        )
