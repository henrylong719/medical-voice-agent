"""
Slot computation engine.

The core scheduling algorithm that answers: "When can this doctor see a patient?"

The formula:
    Available slots = Theoretical slots − Booked appointments − Doctor blocks

Two entry points:
    find_slots_for_specialty()  → "When can I see a cardiologist?" (triage flow)
    find_slots_for_doctor()     → "When can I see Dr. Chen?" (reschedule flow)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import cast

from app.config import settings
from app.supabase_client import supabase
from app.models.slot import SlotDict
from app.models.db_rows import (
    AvailabilityTemplateRow,
    DoctorRow,
    DoctorSpecialtyRow,
    SpecialtyRow,
    TimeRangeRow,
)
from app.services.time_utils import (
    Bucket,
    CLINIC_TZ,
    day_range_to_utc,
    format_date_for_voice,
    format_for_voice,
    is_in_bucket,
    now_utc,
    parse_preferred_day,
    parse_time_bucket,
)


# Phrases that mean "just give me the soonest opening"
NEXT_AVAILABLE_ALIASES = {
    "next available", "next available day", "next available date",
    "soonest", "earliest", "earliest available", "first available",
    "as soon as possible", "asap",
}

# Day-of-week enum values in our DB → Python weekday numbers
DAY_TO_WEEKDAY: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


@dataclass
class Slot:
    """A single available appointment slot (times stored in UTC)."""
    doctor_id: str
    doctor_name: str
    specialty_id: str
    specialty_name: str
    start_at: datetime  # UTC
    end_at: datetime    # UTC

    def to_dict(self) -> SlotDict:
        """Convert to a dict with voice-friendly labels."""
        return {
            "doctor_id": self.doctor_id,
            "doctor_name": self.doctor_name,
            "specialty_id": self.specialty_id,
            "specialty_name": self.specialty_name,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "label": format_for_voice(self.start_at),
            "date_label": format_date_for_voice(self.start_at),
        }


# ============================================================
# PUBLIC API
# ============================================================

def find_slots_for_specialty(
    specialty_id: str,
    preferred_day: str | None = None,
    preferred_time: str | None = None,
    max_results: int = 10,
) -> list[SlotDict]:
    """
    Find available slots across all active doctors with a given specialty.

    Used during triage: patient describes symptoms → specialty matched →
    "Here are the next available cardiology appointments."
    """
    doctors = _get_doctors_for_specialty(specialty_id)
    if not doctors:
        return []

    all_slots: list[Slot] = []
    for doctor in doctors:
        slots = _compute_available_slots(
            doctor_id=doctor["doctor_id"],
            doctor_name=doctor["doctors"]["full_name"],
            specialty_id=specialty_id,
            specialty_name=doctor["specialties"]["name"],
            preferred_day=preferred_day,
            preferred_time=preferred_time,
        )
        all_slots.extend(slots)

    all_slots.sort(key=lambda s: s.start_at)
    return [s.to_dict() for s in all_slots[:max_results]]


def find_slots_for_doctor(
    doctor_id: str,
    specialty_id: str,
    preferred_day: str | None = None,
    preferred_time: str | None = None,
    max_results: int = 10,
) -> list[SlotDict]:
    """
    Find available slots for a specific doctor.

    Used when rescheduling: "I want to reschedule my appointment with Dr. Chen."
    """
    doctor = _get_doctor(doctor_id)
    if not doctor:
        return []

    specialty = _get_specialty(specialty_id)
    spec_name = specialty["name"] if specialty else "Unknown"

    slots = _compute_available_slots(
        doctor_id=doctor_id,
        doctor_name=doctor["full_name"],
        specialty_id=specialty_id,
        specialty_name=spec_name,
        preferred_day=preferred_day,
        preferred_time=preferred_time,
    )

    slots.sort(key=lambda s: s.start_at)
    return [s.to_dict() for s in slots[:max_results]]


# ============================================================
# CORE ALGORITHM
# ============================================================

def _compute_available_slots(
    doctor_id: str,
    doctor_name: str,
    specialty_id: str,
    specialty_name: str,
    preferred_day: str | None,
    preferred_time: str | None,
) -> list[Slot]:
    """
    Core slot computation for a single doctor.

    1. Parse preferences → search window
    2. Clamp to [now, horizon]
    3. Generate theoretical slots from weekly templates
    4. Subtract booked appointments and blocks
    """
    now = now_utc()
    horizon = now + timedelta(days=settings.scheduling_horizon_days)
    bucket = parse_time_bucket(preferred_time)

    # Determine search window
    day_raw = (preferred_day or "").strip().lower()
    if day_raw in NEXT_AVAILABLE_ALIASES:
        utc_start, utc_end = now, horizon
    else:
        day_range = parse_preferred_day(preferred_day)
        utc_start, utc_end = day_range_to_utc(day_range)

    # Clamp: don't search into the past or beyond the horizon
    if utc_end <= now:
        return []
    utc_start = max(utc_start, now)
    utc_end = min(utc_end, horizon)

    # Step 1: Fetch templates
    templates = _get_availability_templates(doctor_id)
    if not templates:
        return []

    # Step 2: Generate theoretical slots
    theoretical = _generate_theoretical_slots(
        doctor_id=doctor_id,
        doctor_name=doctor_name,
        specialty_id=specialty_id,
        specialty_name=specialty_name,
        templates=templates,
        utc_start=utc_start,
        utc_end=utc_end,
        bucket=bucket,
    )

    # Step 3: Fetch conflicts
    booked = _get_booked_slots(doctor_id, utc_start, utc_end)
    blocks = _get_doctor_blocks(doctor_id, utc_start, utc_end)

    # Step 4: Subtract conflicts
    return _subtract_conflicts(theoretical, booked, blocks)


# ============================================================
# DATA ACCESS
# ============================================================

def _get_doctors_for_specialty(specialty_id: str) -> list[DoctorSpecialtyRow]:
    """Fetch active doctors who practice a given specialty."""
    result = (
        supabase.table("doctor_specialties")
        .select("doctor_id, doctors(full_name, is_active), specialties(name)")
        .eq("specialty_id", specialty_id)
        .execute()
    )
    rows = cast(list[DoctorSpecialtyRow], result.data or [])
    return [row for row in rows if row["doctors"]["is_active"]]


def _get_doctor(doctor_id: str) -> DoctorRow | None:
    """Fetch a single active doctor by ID."""
    result = (
        supabase.table("doctors")
        .select("id, full_name, is_active")
        .eq("id", doctor_id)
        .eq("is_active", True)
        .execute()
    )
    rows = cast(list[DoctorRow], result.data or [])
    return rows[0] if rows else None


def _get_specialty(specialty_id: str) -> SpecialtyRow | None:
    """Fetch a single specialty by ID."""
    result = (
        supabase.table("specialties")
        .select("id, name")
        .eq("id", specialty_id)
        .execute()
    )
    rows = cast(list[SpecialtyRow], result.data or [])
    return rows[0] if rows else None


def _get_availability_templates(doctor_id: str) -> list[AvailabilityTemplateRow]:
    """Fetch weekly availability templates for a doctor."""
    result = (
        supabase.table("doctor_availability")
        .select("*")
        .eq("doctor_id", doctor_id)
        .execute()
    )
    return cast(list[AvailabilityTemplateRow], result.data or [])


def _get_booked_slots(
    doctor_id: str,
    utc_start: datetime,
    utc_end: datetime,
) -> set[tuple[datetime, datetime]]:
    """
    Fetch booked appointments as a set of (start, end) tuples.

    Uses a set for O(1) lookups during conflict checking.
    Only includes non-cancelled appointments.
    """
    result = (
        supabase.table("appointments")
        .select("start_at, end_at")
        .eq("doctor_id", doctor_id)
        .neq("status", "cancelled")
        .gte("start_at", utc_start.isoformat())
        .lt("start_at", utc_end.isoformat())
        .execute()
    )
    return {
        (
            datetime.fromisoformat(row["start_at"]),
            datetime.fromisoformat(row["end_at"]),
        )
        for row in cast(list[TimeRangeRow], result.data or [])
    }


def _get_doctor_blocks(
    doctor_id: str,
    utc_start: datetime,
    utc_end: datetime,
) -> list[tuple[datetime, datetime]]:
    """
    Fetch doctor blocks (time off) that overlap with the search window.

    Uses overlapping range query: block starts before window ends
    AND block ends after window starts.
    """
    result = (
        supabase.table("doctor_blocks")
        .select("start_at, end_at")
        .eq("doctor_id", doctor_id)
        .lt("start_at", utc_end.isoformat())
        .gt("end_at", utc_start.isoformat())
        .execute()
    )
    return [
        (
            datetime.fromisoformat(row["start_at"]),
            datetime.fromisoformat(row["end_at"]),
        )
        for row in cast(list[TimeRangeRow], result.data or [])
    ]


# ============================================================
# SLOT GENERATION & CONFLICT RESOLUTION
# ============================================================

def _generate_theoretical_slots(
    doctor_id: str,
    doctor_name: str,
    specialty_id: str,
    specialty_name: str,
    templates: list[AvailabilityTemplateRow],
    utc_start: datetime,
    utc_end: datetime,
    bucket: Bucket,
) -> list[Slot]:
    """
    Generate all theoretical slots from weekly templates for a UTC time window.

    Walks through each day, checks if there's a template for that weekday,
    and generates fixed-duration slots. Templates define clinic-local times;
    slots are produced in UTC.
    """
    now = now_utc()
    slots: list[Slot] = []

    # Build lookup: weekday number → list of templates
    templates_by_day: dict[int, list[AvailabilityTemplateRow]] = {}
    for tmpl in templates:
        weekday = DAY_TO_WEEKDAY[tmpl["day_of_week"]]
        templates_by_day.setdefault(weekday, []).append(tmpl)

    # Convert UTC boundaries to clinic-local dates for iteration
    local_start = utc_start.astimezone(CLINIC_TZ).date()
    local_end = utc_end.astimezone(CLINIC_TZ).date() + timedelta(days=1)

    current_date = local_start
    while current_date <= local_end:
        weekday = current_date.weekday()

        if weekday not in templates_by_day:
            current_date += timedelta(days=1)
            continue

        for tmpl in templates_by_day[weekday]:
            tmpl_start = time.fromisoformat(tmpl["start_time"])
            tmpl_end = time.fromisoformat(tmpl["end_time"])
            duration = timedelta(minutes=tmpl["slot_duration_min"])

            # Build clinic-local datetimes, convert to UTC
            window_start = datetime.combine(
                current_date, tmpl_start, tzinfo=CLINIC_TZ,
            ).astimezone(timezone.utc)
            window_end = datetime.combine(
                current_date, tmpl_end, tzinfo=CLINIC_TZ,
            ).astimezone(timezone.utc)

            slot_start = window_start
            while slot_start + duration <= window_end:
                slot_end = slot_start + duration

                if slot_start <= now:
                    slot_start = slot_end
                    continue

                if slot_start < utc_start or slot_start >= utc_end:
                    slot_start = slot_end
                    continue

                if not is_in_bucket(slot_start, bucket):
                    slot_start = slot_end
                    continue

                slots.append(Slot(
                    doctor_id=doctor_id,
                    doctor_name=doctor_name,
                    specialty_id=specialty_id,
                    specialty_name=specialty_name,
                    start_at=slot_start,
                    end_at=slot_end,
                ))

                slot_start = slot_end

        current_date += timedelta(days=1)

    return slots


def _subtract_conflicts(
    slots: list[Slot],
    booked: set[tuple[datetime, datetime]],
    blocks: list[tuple[datetime, datetime]],
) -> list[Slot]:
    """
    Remove slots that overlap with booked appointments or doctor blocks.

    For booked appointments: checks exact (start, end) match in the set
    AND checks for partial overlap (a 60-min appointment blocking two 30-min slots).

    For blocks: checks any time range overlap (a vacation day blocks all slots).
    """
    available: list[Slot] = []

    for slot in slots:
        # Quick check: exact match in booked set (O(1))
        if (slot.start_at, slot.end_at) in booked:
            continue

        # Thorough check: partial overlap with any booked appointment
        booked_conflict = any(
            slot.start_at < booked_end and booked_start < slot.end_at
            for booked_start, booked_end in booked
        )
        if booked_conflict:
            continue

        # Block check: any overlap with doctor blocks
        block_conflict = any(
            slot.start_at < block_end and block_start < slot.end_at
            for block_start, block_end in blocks
        )
        if block_conflict:
            continue

        available.append(slot)

    return available