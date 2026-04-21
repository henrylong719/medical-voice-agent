"""Pydantic models for doctors and availability."""

from pydantic import BaseModel, Field


class DoctorIn(BaseModel):
    """Request model for creating a doctor."""

    full_name: str = Field(min_length=1, description="Doctor's full name")
    email: str | None = None
    phone: str | None = None
    image_url: str | None = None


class AvailabilityIn(BaseModel):
    """Request model for adding a weekly availability template."""

    day_of_week: str = Field(description="e.g. 'monday', 'tuesday'")
    start_time: str = Field(description="e.g. '09:00'")
    end_time: str = Field(description="e.g. '14:00'")
    slot_duration_min: int = Field(default=30, ge=15, le=240)


class DoctorCreateIn(BaseModel):
    """Request model for creating a doctor with specialties and availability."""

    doctor: DoctorIn
    specialty_ids: list[str] = Field(description="List of specialty UUIDs")
    availability: list[AvailabilityIn] = Field(default=[])
