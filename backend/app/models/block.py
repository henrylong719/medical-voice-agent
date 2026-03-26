"""Pydantic models for doctor blocks (time off)."""

from pydantic import BaseModel, Field


class BlockIn(BaseModel):
    """Request model for adding a doctor block (time off)."""
    doctor_id: str
    start_at: str = Field(description="ISO datetime, e.g. '2026-04-10T09:00:00+00:00'")
    end_at: str = Field(description="ISO datetime")
    reason: str | None = None