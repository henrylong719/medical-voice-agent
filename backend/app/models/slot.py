"""Pydantic models and TypedDicts for appointment slots."""

from typing import TypedDict


class SlotDict(TypedDict):
    """Shape of a slot returned by the slot engine."""

    doctor_id: str
    doctor_name: str
    specialty_id: str
    specialty_name: str
    start_at: str
    end_at: str
    label: str
    date_label: str
