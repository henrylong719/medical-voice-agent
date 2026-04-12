"""
TypedDict definitions for Supabase query results.

These describe the shape of rows returned by Supabase queries.
Used with cast() to give the type checker visibility into
untyped dict data from the database client.

Organized by:
  - Single table rows (one table, no joins)
  - Shared nested row fragments (reused across joined queries)
  - Joined query rows — Phase 1 (slot engine, admin routes)
  - Joined query rows — Phase 2 (agent tools)
  - Aggregation types (intermediate structures, not raw DB rows)
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


class PatientLookupRow(TypedDict):
    """Shape for patient lookup queries (identify / register)."""
    id: str
    uin: str
    full_name: str
    phone: str | None
    email: str | None
    allergies: list[str] | None


class SpecialtyListRow(TypedDict):
    """Shape for specialty listing queries (includes description)."""
    id: str
    name: str
    description: str | None


class CreatedAppointmentRow(TypedDict):
    """Shape returned after inserting an appointment."""
    id: str


# ============================================================
# SHARED NESTED ROW FRAGMENTS
# Reusable pieces that appear inside multiple joined queries.
# ============================================================

class NestedDoctorName(TypedDict):
    """Nested doctors(full_name) — used wherever we join doctor name."""
    full_name: str


class NestedSpecialtyName(TypedDict):
    """Nested specialties(name) — used wherever we join specialty name."""
    name: str


# ============================================================
# JOINED QUERY ROWS — PHASE 1 (slot engine, admin routes)
# ============================================================

class DoctorSpecialtyDoctorRow(TypedDict):
    """Nested doctor fields in a doctor_specialties join."""
    full_name: str
    is_active: bool


class DoctorSpecialtyRow(TypedDict):
    """Shape of a doctor_specialties row with joined doctor and specialty."""
    doctor_id: str
    doctors: DoctorSpecialtyDoctorRow
    specialties: NestedSpecialtyName


# ============================================================
# JOINED QUERY ROWS — PHASE 2 (agent tools)
# ============================================================

class TriageMatchRow(TypedDict):
    """Shape of a symptom_specialty_map row with joined specialty name."""
    symptom: str
    weight: float | int
    follow_up_questions: list[str] | None
    specialty_id: str
    specialties: NestedSpecialtyName


class FindAppointmentRow(TypedDict):
    """Shape for listing scheduled appointments with doctor and specialty."""
    id: str
    start_at: str
    end_at: str
    status: str
    reason: str | None
    doctors: NestedDoctorName
    specialties: NestedSpecialtyName


class RescheduleAppointmentRow(TypedDict):
    """Shape for reschedule lookups (needs doctor_id and specialty_id for re-search)."""
    id: str
    patient_id: str
    doctor_id: str
    specialty_id: str
    start_at: str
    end_at: str
    status: str
    reason: str | None
    doctors: NestedDoctorName
    specialties: NestedSpecialtyName


class CancelAppointmentRow(TypedDict):
    """Shape for cancel lookups (needs start_at for confirmation message)."""
    id: str
    start_at: str
    status: str
    doctors: NestedDoctorName
    specialties: NestedSpecialtyName


# ============================================================
# AGGREGATION TYPES
# Not raw DB rows — intermediate structures built in application code.
# ============================================================

class TriageScore(TypedDict):
    """Aggregated score for one specialty during triage."""
    name: str
    total_weight: float
    matched_symptoms: list[str]
    follow_up_questions: list[str]
