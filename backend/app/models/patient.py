"""Pydantic models for patients."""

from pydantic import BaseModel, Field


class PatientIn(BaseModel):
    """Request model for creating/registering a patient."""
    uin: str = Field(pattern=r"^\d{9}$", description="9-digit university ID")
    full_name: str = Field(min_length=1)
    phone: str | None = None
    email: str | None = None
    allergies: list[str] = Field(default=[])