"""Pydantic models for patients."""

from typing import Literal

from pydantic import BaseModel, Field


IdentifierType = Literal["mrn", "passport", "drivers_license", "external_patient_id"]


class PatientIn(BaseModel):
    """Request model for creating/registering a patient."""

    full_name: str = Field(min_length=1)
    date_of_birth: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date of birth in YYYY-MM-DD format",
    )
    phone: str | None = None
    email: str | None = None
    allergies: list[str] = Field(default_factory=list)


class PatientSearchIn(BaseModel):
    """Request model for demographic patient search."""

    full_name: str = Field(min_length=1)
    date_of_birth: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date of birth in YYYY-MM-DD format",
    )
    phone: str | None = None


class PatientIdentifierIn(BaseModel):
    """Request model for attaching a patient identifier."""

    identifier_type: IdentifierType
    identifier_value: str = Field(min_length=1)
    issuing_country: str | None = None
    is_primary: bool = False
