"""Pydantic models for specialties."""

from pydantic import BaseModel


class SpecialtyOut(BaseModel):
    """Response model for a specialty."""

    id: str
    name: str
    description: str | None = None
