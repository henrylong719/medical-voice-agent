"""
TypedDict definitions for Supabase query results.

These describe the shape of rows returned by Supabase queries.
Used with cast() to give the type checker visibility into
untyped dict data from the database client.
"""

from typing import TypedDict


# ============================================================
# SINGLE TABLE ROWS
# ============================================================

class DoctorRow(TypedDict):
    """Shape of a row from the doctors table."""
    id: str
    full_name: str
    is_active: bool


class SpecialtyRow(TypedDict):
    """Shape of a row from the specialties table."""
    id: str
    name: str


class AvailabilityTemplateRow(TypedDict):
    """Shape of a row from the doctor_availability table."""
    day_of_week: str
    start_time: str
    end_time: str
    slot_duration_min: int


class TimeRangeRow(TypedDict):
    """Shape for any query that returns start_at/end_at pairs."""
    start_at: str
    end_at: str


# ============================================================
# JOINED QUERY ROWS
# These match the shape of Supabase queries with nested selects.
# e.g. .select("doctor_id, doctors(full_name, is_active)")
# ============================================================

class DoctorSpecialtyDoctorRow(TypedDict):
    """Nested doctor fields in a doctor_specialties join."""
    full_name: str
    is_active: bool


class DoctorSpecialtySpecialtyRow(TypedDict):
    """Nested specialty fields in a doctor_specialties join."""
    name: str


class DoctorSpecialtyRow(TypedDict):
    """Shape of a doctor_specialties row with joined doctor and specialty."""
    doctor_id: str
    doctors: DoctorSpecialtyDoctorRow
    specialties: DoctorSpecialtySpecialtyRow