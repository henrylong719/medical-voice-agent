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
import re
from typing import cast

from app.core.config import settings
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
    "next available",
    "next available day",
    "next available date",
    "soonest",
    "earliest",
    "earliest available",
    "first available",
    "as soon as possible",
    "asap",
    "soonest available",
    "no preference",
    "flexible",
    "im flexible",
    "i'm flexible",
    "any day",
    "any day works",
    "whenever",
    "whatever is available",
    "whatever is open",
    "whatever's available",
    "whatever's open",
    "whats available",
    "what's available",
    "anything available",
    "anything open",
}

# Day-of-week enum values in our DB → Python weekday numbers
DAY_TO_WEEKDAY: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass
class Slot:
    """A single available appointment slot (times stored in UTC)."""

    doctor_id: str
    doctor_name: str
    specialty_id: str
    specialty_name: str
    start_at: datetime  # UTC
    end_at: datetime  # UTC

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


def _normalize_preferred_day_text(preferred_day: str | None) -> str:
    """Normalize free-text day preferences for alias detection."""
    normalized = (preferred_day or "").strip().lower().replace("’", "'")
    normalized = re.sub(r"[^a-z0-9\s'/:-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _looks_like_specific_day_request(normalized: str) -> bool:
    """Return True when the text contains a concrete day or date hint."""
    if not normalized:
        return False

    specific_markers = (
        "today",
        "tomorrow",
        "tmr",
        "tommorow",
        "this week",
        "next week",
        "weekend",
        "mon",
        "monday",
        "tue",
        "tues",
        "tuesday",
        "wed",
        "weds",
        "wednesday",
        "thu",
        "thur",
        "thurs",
        "thursday",
        "fri",
        "friday",
        "sat",
        "saturday",
        "sun",
        "sunday",
        "jan",
        "january",
        "feb",
        "february",
        "mar",
        "march",
        "apr",
        "april",
        "may",
        "jun",
        "june",
        "jul",
        "july",
        "aug",
        "august",
        "sep",
        "sept",
        "september",
        "oct",
        "october",
        "nov",
        "november",
        "dec",
        "december",
    )
    if any(
        re.search(rf"\b{re.escape(marker)}\b", normalized)
        for marker in specific_markers
    ):
        return True

    return bool(
        re.search(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", normalized)
        or re.search(r"\b\d+\s*weeks?\b", normalized)
    )


def _is_next_available_preference(preferred_day: str | None) -> bool:
    """Return True when the patient is asking for the soonest open day."""
    normalized = _normalize_preferred_day_text(preferred_day)
    if not normalized:
        return False
    if normalized in NEXT_AVAILABLE_ALIASES:
        return True
    if _looks_like_specific_day_request(normalized):
        return False
    return any(
        marker in normalized
        for marker in (
            "soonest",
            "earliest",
            "as soon as possible",
            "asap",
            "flexible",
            "no preference",
            "any day",
            "whenever",
            "available",
            "open",
        )
    )


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


def validate_slot_selection(
    doctor_id: str,
    specialty_id: str,
    start_at: str,
    end_at: str,
    *,
    exclude_appointment_id: str | None = None,
) -> str | None:
    """Validate that a specific slot is still bookable right now."""
    try:
        requested_start = datetime.fromisoformat(start_at)
        requested_end = datetime.fromisoformat(end_at)
    except ValueError:
        return (
            "That appointment time is invalid. Please choose one of the available "
            "options."
        )

    if (
        requested_start.tzinfo is None
        or requested_start.utcoffset() is None
        or requested_end.tzinfo is None
        or requested_end.utcoffset() is None
    ):
        return (
            "That appointment time is invalid. Please choose one of the available "
            "options."
        )

    requested_start = requested_start.astimezone(timezone.utc)
    requested_end = requested_end.astimezone(timezone.utc)

    if requested_end <= requested_start:
        return (
            "That appointment time is invalid. Please choose one of the available "
            "options."
        )

    now = now_utc()
    horizon = now + timedelta(days=settings.SCHEDULING_HORIZON_DAYS)
    if requested_start <= now or requested_end > horizon:
        return (
            "That appointment time is no longer bookable. Please choose another slot."
        )

    doctor = _get_doctor(doctor_id)
    if not doctor:
        return "That doctor is not available for booking. Please choose another slot."

    specialty = _get_specialty(specialty_id)
    if not specialty:
        return (
            "That specialty is not available for booking. Please choose another slot."
        )

    if not _doctor_has_specialty(doctor_id, specialty_id):
        return (
            "That doctor is not available for the selected specialty. Please choose "
            "another slot."
        )

    local_day = requested_start.astimezone(CLINIC_TZ).date()
    window_start = datetime.combine(local_day, time.min, tzinfo=CLINIC_TZ).astimezone(
        timezone.utc
    )
    window_end = datetime.combine(
        local_day + timedelta(days=1),
        time.min,
        tzinfo=CLINIC_TZ,
    ).astimezone(timezone.utc)

    theoretical = _generate_theoretical_slots(
        doctor_id=doctor_id,
        doctor_name=doctor["full_name"],
        specialty_id=specialty_id,
        specialty_name=specialty["name"],
        templates=_get_availability_templates(doctor_id),
        utc_start=window_start,
        utc_end=window_end,
        bucket="any",
    )

    requested_slot = next(
        (
            slot
            for slot in theoretical
            if slot.start_at == requested_start and slot.end_at == requested_end
        ),
        None,
    )
    if requested_slot is None:
        return (
            "That time does not match the doctor's current availability. Please choose "
            "one of the offered slots."
        )

    booked = _get_booked_slots(
        doctor_id,
        window_start,
        window_end,
        exclude_appointment_id=exclude_appointment_id,
    )
    blocks = _get_doctor_blocks(doctor_id, window_start, window_end)
    if not _subtract_conflicts([requested_slot], booked, blocks):
        return "That slot is no longer available. Please choose another available time."

    return None


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
    horizon = now + timedelta(days=settings.SCHEDULING_HORIZON_DAYS)
    bucket = parse_time_bucket(preferred_time)

    # Determine search window
    day_raw = _normalize_preferred_day_text(preferred_day)
    if not day_raw or _is_next_available_preference(day_raw):
        # No preference or "next available" → search the full horizon
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


def _doctor_has_specialty(doctor_id: str, specialty_id: str) -> bool:
    """Return whether an active doctor is linked to the given specialty."""
    result = (
        supabase.table("doctor_specialties")
        .select("doctor_id")
        .eq("doctor_id", doctor_id)
        .eq("specialty_id", specialty_id)
        .execute()
    )
    return bool(result.data)


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
    exclude_appointment_id: str | None = None,
) -> set[tuple[datetime, datetime]]:
    """
    Fetch booked appointments as a set of (start, end) tuples.

    Uses a set for O(1) lookups during conflict checking.
    Only includes non-cancelled appointments that overlap the search window.
    """
    query = (
        supabase.table("appointments")
        .select("start_at, end_at")
        .eq("doctor_id", doctor_id)
        .neq("status", "cancelled")
        .lt("start_at", utc_end.isoformat())
        .gt("end_at", utc_start.isoformat())
    )
    if exclude_appointment_id:
        query = query.neq("id", exclude_appointment_id)

    result = query.execute()
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
                current_date,
                tmpl_start,
                tzinfo=CLINIC_TZ,
            ).astimezone(timezone.utc)
            window_end = datetime.combine(
                current_date,
                tmpl_end,
                tzinfo=CLINIC_TZ,
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

                slots.append(
                    Slot(
                        doctor_id=doctor_id,
                        doctor_name=doctor_name,
                        specialty_id=specialty_id,
                        specialty_name=specialty_name,
                        start_at=slot_start,
                        end_at=slot_end,
                    )
                )

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
