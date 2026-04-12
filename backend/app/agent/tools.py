"""
Agent tools for the medical scheduling assistant.

Each tool wraps a Phase 1 service function with:
  - A Pydantic input schema (tells the LLM what arguments to provide)
  - A descriptive docstring (tells the LLM WHEN to use this tool)
  - A handler that calls Supabase directly (no HTTP round-trips)

The LLM reads the tool name, description, and schema to decide which
tool to call. Better descriptions → better tool selection.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Literal, cast

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.supabase_client import supabase
from app.services.slot_engine import (
    find_slots_for_specialty,
    find_slots_for_doctor,
    validate_slot_selection,
)
from app.services.time_utils import format_for_voice, now_utc
from app.services.rag_retriever import retrieve_medical_knowledge
from app.models.db_rows import (
    CancelAppointmentRow,
    CreatedAppointmentRow,
    FindAppointmentRow,
    PatientIdentifierRow,
    PatientLookupRow,
    RescheduleAppointmentRow,
    SpecialtyListRow,
    TriageMatchRow,
    TriageScore,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# INPUT SCHEMAS
# ============================================================
# These Pydantic models define what the LLM must provide when
# calling each tool. The Field descriptions are critical — the
# LLM reads them to understand what values to pass.
# ============================================================

IdentifierType = Literal["mrn", "passport", "drivers_license", "external_patient_id"]

class FindPatientByIdentifierInput(BaseModel):
    """Input for patient lookup by a strong identifier."""
    identifier_type: IdentifierType = Field(
        description=(
            "Type of identifier provided by the patient: mrn, passport, "
            "drivers_license, or external_patient_id"
        )
    )
    identifier_value: str = Field(
        description="Identifier value exactly as the patient provided it"
    )


class FindPatientsByDemographicsInput(BaseModel):
    """Input for patient lookup by demographics."""
    full_name: str = Field(description="Patient's full name")
    date_of_birth: str = Field(
        description=(
            "Patient's date of birth. Accept common formats like "
            "YYYY-MM-DD, MM-DD-YYYY, MM/DD/YYYY, or month-name dates. "
            "The backend normalizes this to YYYY-MM-DD."
        )
    )
    phone: str = Field(
        default="",
        description=(
            "Optional phone number to narrow demographic matches. If omitted, "
            "use an empty string."
        ),
    )


class RegisterPatientInput(BaseModel):
    """Input for registering a new patient."""
    full_name: str = Field(description="Patient's full name")
    date_of_birth: str = Field(
        description=(
            "Patient's date of birth. Accept common formats like "
            "YYYY-MM-DD, MM-DD-YYYY, MM/DD/YYYY, or month-name dates. "
            "The backend normalizes this to YYYY-MM-DD."
        )
    )
    phone: str = Field(
        description=(
            "Patient's phone number. This is required for new patient "
            "registration. Accept any non-empty phone number string exactly "
            "as the patient provides it; do not require a specific format or "
            "area code."
        ),
    )
    email: str | None = Field(
        default=None,
        description="Optional patient email address, if they provide one",
    )


class TriageSymptomsInput(BaseModel):
    """Input for symptom-based triage."""
    symptoms: list[str] = Field(
        description=(
            "List of individual symptoms the patient described. "
            "Break compound descriptions into separate symptoms. "
            "Example: ['chest pain', 'shortness of breath', 'dizziness']"
        )
    )
    description: str = Field(
        default="",
        description=(
            "The patient's full, natural language description of how they feel. "
            "Keep the original wording as close as possible. "
            "Example: 'I have been getting sharp pains behind my eyes and I "
            "sometimes see flashing lights before it starts'"
        ),
    )


class FindSlotsInput(BaseModel):
    """Input for finding available appointment slots."""
    specialty_id: str = Field(description="UUID of the specialty to search")
    preferred_day: str | None = Field(
        default=None,
        description=(
            "When the patient wants the appointment. Supports natural language: "
            "'tomorrow', 'next tuesday', 'this week', 'feb 24', 'next available'. "
            "Leave empty to search the full scheduling horizon."
        ),
    )
    preferred_time: str | None = Field(
        default=None,
        description=(
            "Time of day preference: 'morning', 'afternoon', or leave empty for any time"
        ),
    )


class BookAppointmentInput(BaseModel):
    """Input for booking an appointment."""
    patient_id: str = Field(description="UUID of the patient")
    doctor_id: str = Field(description="UUID of the doctor (from find_slots results)")
    specialty_id: str = Field(description="UUID of the specialty")
    start_at: str = Field(description="Slot start time in ISO format (from find_slots results)")
    end_at: str = Field(description="Slot end time in ISO format (from find_slots results)")
    reason: str | None = Field(
        default=None,
        description="Brief reason for the visit based on the patient's symptoms",
    )


class FindAppointmentInput(BaseModel):
    """Input for looking up existing appointments."""
    patient_id: str = Field(description="UUID of the patient")
    doctor_name: str | None = Field(
        default=None,
        description="Optional doctor name to filter by (partial match supported)",
    )
    specialty_name: str | None = Field(
        default=None,
        description="Optional specialty name to filter by (partial match supported)",
    )


class RescheduleInput(BaseModel):
    """Input for rescheduling an appointment."""
    appointment_id: str = Field(description="UUID of the appointment to reschedule")
    patient_id: str = Field(description="UUID of the patient who owns the appointment")
    preferred_day: str | None = Field(
        default=None,
        description="When the patient wants the new appointment (natural language, for previewing options)",
    )
    preferred_time: str | None = Field(
        default=None,
        description="Time of day preference: 'morning', 'afternoon', or empty for any, for previewing options",
    )
    new_doctor_id: str | None = Field(
        default=None,
        description="Doctor UUID for the confirmed new slot when finalizing the reschedule",
    )
    new_specialty_id: str | None = Field(
        default=None,
        description="Specialty UUID for the confirmed new slot when finalizing the reschedule",
    )
    new_start_at: str | None = Field(
        default=None,
        description="Confirmed new slot start time in ISO format when finalizing the reschedule",
    )
    new_end_at: str | None = Field(
        default=None,
        description="Confirmed new slot end time in ISO format when finalizing the reschedule",
    )


class CancelAppointmentInput(BaseModel):
    """Input for cancelling an appointment."""
    patient_id: str = Field(description="UUID of the patient who owns the appointment")
    appointment_id: str = Field(description="UUID of the appointment to cancel")


# ============================================================
# TOOLS
# ============================================================
# Each @tool function is what the LLM can call. The docstring
# is the tool description the LLM reads. Return strings —
# the LLM needs text it can reason about, not raw dicts.
# ============================================================


def _format_doctor_name_for_voice(full_name: str) -> str:
    """Add a Dr. title unless the source name already includes one."""
    normalized = full_name.strip()
    lower = normalized.lower()
    if lower.startswith("dr. ") or lower.startswith("dr "):
        return normalized
    return f"Dr. {normalized}"


def _format_preference_suffix(
    preferred_day: str | None = None,
    preferred_time: str | None = None,
) -> str:
    """Render natural-language slot filters for tool responses."""
    phrases: list[str] = []
    if preferred_day:
        phrases.append(f"for {preferred_day}")
    if preferred_time:
        phrases.append(f"in the {preferred_time}")
    if not phrases:
        return ""
    return f" {' '.join(phrases)}"


def _coerce_rpc_payload(data: object) -> dict[str, object]:
    """Normalize Supabase RPC responses into a dict payload."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def _mask_phone(phone: str | None) -> str | None:
    """Return only the last four digits of a phone number when possible."""
    if phone is None:
        return None
    digits = "".join(char for char in phone if char.isdigit())
    if len(digits) >= 4:
        return digits[-4:]
    normalized = phone.strip()
    return normalized or None


def _mask_identifier(value: str) -> str:
    """Mask an identifier value so only the last four characters are exposed."""
    normalized = value.strip()
    if len(normalized) <= 4:
        return normalized
    return f"{'*' * (len(normalized) - 4)}{normalized[-4:]}"


def _normalize_date_of_birth(date_of_birth: str) -> str | None:
    """Normalize common DOB inputs to ISO YYYY-MM-DD."""
    normalized = date_of_birth.strip()
    if not normalized:
        return None

    cleaned = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", normalized, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for fmt in (
        "%Y-%m-%d",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m-%d-%y",
        "%m/%d/%y",
        "%B %d %Y",
        "%b %d %Y",
    ):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue

    return None


def _serialize_patient(patient: PatientLookupRow) -> dict[str, object]:
    """Convert a patient row into a tool-friendly payload."""
    return {
        "patient_id": patient["id"],
        "full_name": patient["full_name"],
        "date_of_birth": patient["date_of_birth"],
        "phone_last4": _mask_phone(patient.get("phone")),
        "email": patient.get("email"),
        "allergies": patient.get("allergies") or [],
    }


def _json_tool_response(
    *,
    status: str,
    message: str,
    patient: dict[str, object] | None = None,
    candidates: list[dict[str, object]] | None = None,
    lookup_method: str | None = None,
    matched_identifier: dict[str, object] | None = None,
) -> str:
    """Return a structured JSON payload for the LLM and node parser."""
    payload: dict[str, object] = {
        "status": status,
        "message": message,
    }
    if patient is not None:
        payload["patient"] = patient
    if candidates is not None:
        payload["candidates"] = candidates
        payload["match_count"] = len(candidates)
    if lookup_method is not None:
        payload["lookup_method"] = lookup_method
    if matched_identifier is not None:
        payload["matched_identifier"] = matched_identifier
    return json.dumps(payload)


def _fetch_patient(patient_id: str) -> PatientLookupRow | None:
    """Fetch a single patient row by UUID."""
    result = (
        supabase.table("patients")
        .select("id, full_name, date_of_birth, phone, email, allergies")
        .eq("id", patient_id)
        .execute()
    )
    if not result.data:
        return None
    return cast(list[PatientLookupRow], result.data)[0]


@tool(args_schema=FindPatientByIdentifierInput)
def find_patient_by_identifier(identifier_type: IdentifierType, identifier_value: str) -> str:
    """Look up a returning patient by a strong identifier.

    Use this as a fallback after demographic lookup could not find a
    single clear patient. Strong identifiers include MRN, passport
    number, driver's license number, or another clinic patient number.
    Returns one of: single_match, multiple_matches, or no_match.
    """
    normalized_value = identifier_value.strip()
    if not normalized_value:
        return _json_tool_response(
            status="error",
            message="Identifier value missing. Ask the patient to repeat it before searching.",
            lookup_method="identifier",
        )

    result = (
        supabase.table("patient_identifiers")
        .select("patient_id, identifier_type, identifier_value, issuing_country, is_primary")
        .eq("identifier_type", identifier_type)
        .eq("identifier_value", normalized_value)
        .execute()
    )

    if not result.data:
        return _json_tool_response(
            status="no_match",
            message=(
                "No patient matched that identifier. If name, date of birth, and "
                "phone have already been tried, offer registration or staff help "
                "instead of guessing."
            ),
            lookup_method="identifier",
        )

    matches = cast(list[PatientIdentifierRow], result.data)
    patients: list[PatientLookupRow] = []
    seen_ids: set[str] = set()
    for match in matches:
        patient_id = match["patient_id"]
        if patient_id in seen_ids:
            continue
        seen_ids.add(patient_id)
        patient = _fetch_patient(patient_id)
        if patient is not None:
            patients.append(patient)

    matched_identifier: dict[str, object] = {
        "identifier_type": identifier_type,
        "identifier_value_masked": _mask_identifier(normalized_value),
    }
    if not patients:
        return _json_tool_response(
            status="no_match",
            message="The identifier exists, but no active patient record was found.",
            lookup_method="identifier",
            matched_identifier=matched_identifier,
        )

    if len(patients) == 1:
        return _json_tool_response(
            status="single_match",
            message=(
                "One patient matched that identifier. Confirm the patient's name "
                "and date of birth before proceeding."
            ),
            patient=_serialize_patient(patients[0]),
            lookup_method="identifier",
            matched_identifier=matched_identifier,
        )

    return _json_tool_response(
        status="multiple_matches",
        message=(
            "Multiple patient records matched that identifier. Do not guess. "
            "Hand off to staff for manual verification."
        ),
        candidates=[_serialize_patient(patient) for patient in patients],
        lookup_method="identifier",
        matched_identifier=matched_identifier,
    )


@tool(args_schema=FindPatientsByDemographicsInput)
def find_patients_by_demographics(
    full_name: str,
    date_of_birth: str,
    phone: str = "",
) -> str:
    """Look up a patient by full name and date of birth.

    Use this as the first lookup step for returning patients.
    If multiple matches remain, ask for a phone number and search again.
    Only ask for a stronger identifier if demographics plus phone still
    cannot isolate one patient record.
    """
    normalized_name = full_name.strip()
    normalized_dob = _normalize_date_of_birth(date_of_birth)
    normalized_phone = phone.strip()

    if not normalized_name or not normalized_dob:
        return _json_tool_response(
            status="error",
            message=(
                "Full name and a valid date of birth are both required for "
                "demographic lookup. If needed, ask the patient to repeat "
                "their birth date."
            ),
            lookup_method="demographics",
        )

    query = (
        supabase.table("patients")
        .select("id, full_name, date_of_birth, phone, email, allergies")
        .eq("full_name", normalized_name)
        .eq("date_of_birth", normalized_dob)
    )
    if normalized_phone:
        query = query.eq("phone", normalized_phone)

    result = query.execute()
    patients = cast(list[PatientLookupRow], result.data or [])

    if not patients:
        return _json_tool_response(
            status="no_match",
            message=(
                "No patient matched that name and date of birth. Ask for a phone "
                "number if you do not have one yet. If that still fails, ask for "
                "a stronger identifier or offer registration if this may be their "
                "first visit."
            ),
            lookup_method="demographics",
        )

    if len(patients) == 1:
        return _json_tool_response(
            status="single_match",
            message=(
                "One patient matched those demographics. Confirm the patient's "
                "name and date of birth before proceeding."
            ),
            patient=_serialize_patient(patients[0]),
            lookup_method="demographics",
        )

    follow_up = (
        "Multiple patients matched those demographics even after using the phone "
        "number. Ask for a stronger identifier such as MRN, passport number, "
        "driver's license number, or another clinic patient number. Do not guess."
        if normalized_phone
        else (
            "Multiple patients matched those demographics. Ask for a phone number "
            "first, then try demographics again before asking for a stronger "
            "identifier."
        )
    )
    return _json_tool_response(
        status="multiple_matches",
        message=follow_up,
        candidates=[_serialize_patient(patient) for patient in patients],
        lookup_method="demographics",
    )


@tool(args_schema=RegisterPatientInput)
def register_patient(
    full_name: str,
    date_of_birth: str,
    phone: str,
    email: str | None = None,
) -> str:
    """Register a new patient who doesn't exist in the system yet.

    Use this only after the patient confirms this is their first visit or
    after lookup confirms there is no matching patient record.
    Collect the patient's full name, date of birth, and phone number first.
    Accept any non-empty phone number string the patient provides.
    Only call this AFTER reading the phone number back slowly and confirming
    with the patient that it is correct.
    """
    normalized_name = full_name.strip()
    normalized_dob = _normalize_date_of_birth(date_of_birth)
    normalized_phone = phone.strip()
    normalized_email = email.strip() if email else None

    if not normalized_name or not normalized_dob:
        return _json_tool_response(
            status="error",
            message=(
                "Full name and a valid date of birth are required before "
                "registering a patient."
            ),
        )

    if not normalized_phone:
        return _json_tool_response(
            status="error",
            message=(
                "Phone number missing. Ask the patient for their phone number "
                "before registering them."
            ),
        )

    existing = (
        supabase.table("patients")
        .select("id, full_name, date_of_birth, phone, email, allergies")
        .eq("full_name", normalized_name)
        .eq("date_of_birth", normalized_dob)
        .eq("phone", normalized_phone)
        .execute()
    )
    if existing.data:
        patient = cast(list[PatientLookupRow], existing.data)[0]
        return _json_tool_response(
            status="already_exists",
            message=(
                "A patient with the same name, date of birth, and phone number "
                "already exists. Use the lookup tools instead of registering again."
            ),
            patient=_serialize_patient(patient),
        )

    data: dict[str, str] = {
        "full_name": normalized_name,
        "date_of_birth": normalized_dob,
        "phone": normalized_phone,
    }
    if normalized_email:
        data["email"] = normalized_email

    result = supabase.table("patients").insert(data).execute()

    if not result.data:
        return _json_tool_response(
            status="error",
            message="Failed to register patient. Please try again.",
        )

    patient = cast(list[PatientLookupRow], result.data)[0]
    return _json_tool_response(
        status="registered",
        message="Patient registered successfully.",
        patient=_serialize_patient(patient),
    )


@tool(args_schema=TriageSymptomsInput)
def triage_symptoms(symptoms: list[str], description: str = "") -> str:
    """Match patient symptoms to medical specialties using hybrid search.

    Combines two search strategies for best results:
      1. Keyword matching against known symptom-specialty mappings
      2. Semantic (RAG) search against the medical knowledge base

    Takes a list of individual symptoms AND the patient's natural language
    description. Use this after identifying the patient and collecting
    their symptoms.
    """
    if not symptoms and not description:
        return "No symptoms provided. Please ask the patient to describe their symptoms."

    lines: list[str] = []

    # ── Path 1: Keyword matching (precise, high-confidence) ──
    # Searches symptom_specialty_map using ilike for exact/partial
    # keyword matches. Works great when patients use clinical terms
    # like "chest pain" or "migraine."
    all_matches: list[TriageMatchRow] = []

    for symptom in symptoms:
        result = (
            supabase.table("symptom_specialty_map")
            .select("symptom, weight, follow_up_questions, specialty_id, specialties(name)")
            .ilike("symptom", f"%{symptom}%")
            .execute()
        )
        if result.data:
            all_matches.extend(cast(list[TriageMatchRow], result.data))

    if all_matches:
        # Aggregate: group by specialty, sum weights, collect follow-up questions
        specialty_scores: dict[str, TriageScore] = {}

        for match in all_matches:
            spec_id = match["specialty_id"]
            spec_name = match["specialties"]["name"]

            if spec_id not in specialty_scores:
                specialty_scores[spec_id] = {
                    "name": spec_name,
                    "total_weight": 0.0,
                    "matched_symptoms": [],
                    "follow_up_questions": [],
                }

            entry = specialty_scores[spec_id]
            entry["total_weight"] += float(match["weight"])
            entry["matched_symptoms"].append(match["symptom"])

            for q in (match.get("follow_up_questions") or []):
                if q not in entry["follow_up_questions"]:
                    entry["follow_up_questions"].append(q)

        ranked = sorted(
            specialty_scores.items(),
            key=lambda item: item[1]["total_weight"],
            reverse=True,
        )

        lines.append("=== Keyword Matches (from symptom database) ===\n")
        for spec_id, info in ranked:
            matched = ", ".join(info["matched_symptoms"])
            lines.append(
                f"- {info['name']} (ID: {spec_id}): "
                f"score {info['total_weight']:.2f}, "
                f"matched on: {matched}"
            )
            if info["follow_up_questions"]:
                top_questions = info["follow_up_questions"][:2]
                lines.append(f"  Follow-up questions: {'; '.join(top_questions)}")

    # ── Path 2: Semantic search (flexible, handles natural language) ──
    # Embeds the patient's description and finds similar medical
    # knowledge chunks via vector similarity. Catches cases like
    # "elephant sitting on my chest" → Cardiology that keywords miss.
    semantic_query = description if description else ", ".join(symptoms)

    try:
        chunks = retrieve_medical_knowledge(
            query=semantic_query,
            match_count=3,
            match_threshold=0.3,
        )
    except RuntimeError as e:
        # If embedding fails (API key missing, network error), fall back
        # to keyword-only results rather than crashing the whole triage
        chunks = []
        lines.append(f"\n(Semantic search unavailable: {e})")

    if chunks:
        lines.append("\n=== Semantic Matches (from medical knowledge base) ===\n")
        for chunk in chunks:
            metadata = chunk["metadata"]
            specialty_name = metadata.get("specialty_name", "Unknown")
            specialty_id = metadata.get("specialty_id", "N/A")
            similarity = chunk["similarity"]
            category = metadata.get("category", "unknown")

            lines.append(
                f"- {specialty_name} (ID: {specialty_id}): "
                f"similarity {similarity:.2f}, "
                f"category: {category}"
            )
            # Include a snippet of the matched content so the LLM can
            # reason about WHY this specialty was matched
            content_preview = chunk["content"][:200]
            lines.append(f"  Context: {content_preview}...")

    # ── Combine results ──────────────────────────────────────
    if not lines:
        return (
            f"No specialty matches found for symptoms: {', '.join(symptoms)}. "
            "Consider asking the patient to describe their symptoms differently, "
            "or use list_specialties to show available options."
        )

    header = "Hybrid triage results — use BOTH keyword and semantic matches to determine the best specialty.\n"
    return header + "\n".join(lines)


@tool(args_schema=FindSlotsInput)
def find_slots(
    specialty_id: str,
    preferred_day: str | None = None,
    preferred_time: str | None = None,
) -> str:
    """Find available appointment slots for a specialty.

    Searches across all doctors who practice the given specialty.
    Supports natural language day preferences like 'tomorrow', 'next tuesday',
    'this week', or 'next available'. Time preferences: 'morning' or 'afternoon'.
    Returns up to 20 slots across multiple days so you can present available
    days to the patient first, then narrow down by time.
    """
    slots = find_slots_for_specialty(
        specialty_id=specialty_id,
        preferred_day=preferred_day,
        preferred_time=preferred_time,
        max_results=20,
    )

    if not slots:
        pref = ""
        if preferred_day:
            pref += f" on {preferred_day}"
        if preferred_time:
            pref += f" in the {preferred_time}"
        return (
            f"No available slots found{pref}. "
            "Try a different day or time, or ask if the patient is flexible."
        )

    lines = [f"Found {len(slots)} available slot(s):\n"]
    for i, slot in enumerate(slots, 1):
        lines.append(
            f"{i}. {slot['doctor_name']} — {slot['label']} "
            f"(doctor_id: {slot['doctor_id']}, "
            f"start: {slot['start_at']}, end: {slot['end_at']})"
        )

    return "\n".join(lines)


@tool(args_schema=BookAppointmentInput)
def book_appointment(
    patient_id: str,
    doctor_id: str,
    specialty_id: str,
    start_at: str,
    end_at: str,
    reason: str | None = None,
) -> str:
    """Book a confirmed appointment for a patient.

    Use the exact doctor_id, start_at, and end_at values from find_slots results.
    Only book AFTER the patient has confirmed the slot they want.
    """
    validation_error = validate_slot_selection(
        doctor_id=doctor_id,
        specialty_id=specialty_id,
        start_at=start_at,
        end_at=end_at,
    )
    if validation_error:
        return validation_error

    data: dict[str, str] = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "specialty_id": specialty_id,
        "start_at": start_at,
        "end_at": end_at,
        "status": "scheduled",
    }
    if reason:
        data["reason"] = reason

    try:
        result = supabase.table("appointments").insert(data).execute()
    except Exception:
        logger.exception("Failed to book appointment")
        return "Failed to book appointment. Please try again."

    if not result.data:
        return "Failed to book appointment. Please try again."

    appt = cast(list[CreatedAppointmentRow], result.data)[0]
    label = format_for_voice(start_at)

    return (
        f"Appointment booked successfully! "
        f"Appointment ID: {appt['id']}. "
        f"Scheduled for {label}."
    )


@tool(args_schema=FindAppointmentInput)
def find_appointment(
    patient_id: str,
    doctor_name: str | None = None,
    specialty_name: str | None = None,
) -> str:
    """Look up a patient's existing appointments.

    Returns upcoming scheduled appointments. Optionally filter by doctor name.
    Use this when a patient wants to reschedule or cancel an existing appointment.
    """
    query = (
        supabase.table("appointments")
        .select("id, start_at, end_at, status, reason, doctors(full_name), specialties(name)")
        .eq("patient_id", patient_id)
        .eq("status", "scheduled")
        .gte("start_at", now_utc().isoformat())
        .order("start_at")
    )

    result = query.execute()

    if not result.data:
        return "No upcoming appointments found for this patient."

    # Filter by doctor name if provided (case-insensitive partial match)
    appointments = cast(list[FindAppointmentRow], result.data)
    if doctor_name:
        doctor_lower = doctor_name.lower()
        appointments = [
            a for a in appointments
            if doctor_lower in a["doctors"]["full_name"].lower()
        ]
        if not appointments:
            return f"No upcoming appointments found with a doctor matching '{doctor_name}'."

    if specialty_name:
        specialty_lower = specialty_name.lower()
        appointments = [
            a for a in appointments
            if specialty_lower in a["specialties"]["name"].lower()
        ]
        if not appointments:
            return (
                "No upcoming appointments found with a specialty matching "
                f"'{specialty_name}'."
            )

    lines = [f"Found {len(appointments)} upcoming appointment(s):\n"]
    for i, appt in enumerate(appointments, 1):
        label = format_for_voice(appt["start_at"])
        lines.append(
            f"{i}. {appt['doctors']['full_name']} ({appt['specialties']['name']}) — "
            f"{label} "
            f"(appointment_id: {appt['id']})"
        )
        if appt.get("reason"):
            lines.append(f"   Reason: {appt['reason']}")

    return "\n".join(lines)


@tool(args_schema=RescheduleInput)
def reschedule_appointment(
    appointment_id: str,
    patient_id: str,
    preferred_day: str | None = None,
    preferred_time: str | None = None,
    new_doctor_id: str | None = None,
    new_specialty_id: str | None = None,
    new_start_at: str | None = None,
    new_end_at: str | None = None,
) -> str:
    """Preview or finalize a reschedule for an existing appointment.

    Preview mode:
      - Provide only appointment_id, preferred_day, and/or preferred_time.
      - You can call preview mode repeatedly as the patient narrows their
        preferred day or time.
      - The tool keeps the current appointment in place and shows alternatives.

    Finalize mode:
      - Provide appointment_id plus the confirmed new_doctor_id,
        new_specialty_id, new_start_at, and new_end_at from a slot search result.
      - The tool validates the slot and updates the existing appointment in place.
    """
    is_finalize = any(
        value is not None
        for value in (
            new_doctor_id,
            new_specialty_id,
            new_start_at,
            new_end_at,
        )
    )
    has_confirmed_slot = (
        new_doctor_id is not None
        and new_specialty_id is not None
        and new_start_at is not None
        and new_end_at is not None
    )

    if is_finalize and not has_confirmed_slot:
        return (
            "To finalize a reschedule, provide new_doctor_id, new_specialty_id, "
            "new_start_at, and new_end_at together."
        )

    # Fetch the existing appointment
    result = (
        supabase.table("appointments")
        .select(
            "id, patient_id, doctor_id, specialty_id, start_at, end_at, status, "
            "reason, doctors(full_name), specialties(name)"
        )
        .eq("id", appointment_id)
        .eq("patient_id", patient_id)
        .execute()
    )

    if not result.data:
        return f"Appointment {appointment_id} not found for this patient."

    appt = cast(list[RescheduleAppointmentRow], result.data)[0]

    if appt["status"] == "cancelled":
        return "This appointment is already cancelled."

    old_label = format_for_voice(appt["start_at"])
    doctor_name = _format_doctor_name_for_voice(appt["doctors"]["full_name"])

    if has_confirmed_slot:
        assert new_doctor_id is not None
        assert new_specialty_id is not None
        assert new_start_at is not None
        assert new_end_at is not None

        confirmed_doctor_id = new_doctor_id
        confirmed_specialty_id = new_specialty_id
        confirmed_start_at = new_start_at
        confirmed_end_at = new_end_at

        if (
            confirmed_doctor_id == appt["doctor_id"]
            and confirmed_specialty_id == appt["specialty_id"]
            and confirmed_start_at == appt["start_at"]
            and confirmed_end_at == appt["end_at"]
        ):
            return "This appointment is already scheduled for that time."

        validation_error = validate_slot_selection(
            doctor_id=confirmed_doctor_id,
            specialty_id=confirmed_specialty_id,
            start_at=confirmed_start_at,
            end_at=confirmed_end_at,
            exclude_appointment_id=appointment_id,
        )
        if validation_error:
            return f"{validation_error} The original appointment was kept."

        try:
            rpc_result = supabase.rpc(
                "finalize_reschedule_appointment",
                {
                    "p_appointment_id": appointment_id,
                    "p_patient_id": patient_id,
                    "p_new_doctor_id": confirmed_doctor_id,
                    "p_new_specialty_id": confirmed_specialty_id,
                    "p_new_start_at": confirmed_start_at,
                    "p_new_end_at": confirmed_end_at,
                    "p_timezone": settings.timezone,
                },
            ).execute()
        except Exception:
            logger.exception("Failed to reschedule appointment")
            return "Failed to update the appointment. The original appointment was kept."

        payload = _coerce_rpc_payload(rpc_result.data)
        status = payload.get("status")

        if status == "appointment_not_found":
            return f"Appointment {appointment_id} not found for this patient."
        if status == "appointment_cancelled":
            return "This appointment is already cancelled."
        if status == "appointment_not_reschedulable":
            return "This appointment can no longer be rescheduled."
        if status == "same_slot":
            return "This appointment is already scheduled for that time."
        if status == "invalid_doctor_specialty":
            return (
                "That doctor is not available for the selected specialty. "
                "The original appointment was kept."
            )
        if status == "invalid_slot":
            return (
                "That time does not match the doctor's current availability. "
                "Please choose one of the offered slots. The original appointment was kept."
            )
        if status in ("doctor_blocked", "slot_unavailable"):
            return (
                "That slot is no longer available. Please choose another available "
                "time. The original appointment was kept."
            )
        if status != "ok":
            logger.error("Unexpected finalize_reschedule_appointment status: %r", status)
            return "Failed to update the appointment. The original appointment was kept."

        new_label = format_for_voice(confirmed_start_at)
        return (
            "Appointment rescheduled successfully! "
            f"Old appointment on {old_label} was moved to {new_label}. "
            f"Appointment ID: {appointment_id}."
        )

    # Find new slots with the same doctor and specialty
    new_slots = find_slots_for_doctor(
        doctor_id=appt["doctor_id"],
        specialty_id=appt["specialty_id"],
        preferred_day=preferred_day,
        preferred_time=preferred_time,
        max_results=5,
    )
    preference_suffix = _format_preference_suffix(
        preferred_day=preferred_day,
        preferred_time=preferred_time,
    )

    lines = [
        f"Current appointment: {doctor_name} "
        f"({appt['specialties']['name']}) on {old_label} "
        f"(appointment_id: {appt['id']}, specialty_id: {appt['specialty_id']}).\n",
        "The current appointment has NOT been cancelled.\n",
    ]
    if preference_suffix:
        lines.append(f"Search criteria: same doctor{preference_suffix}.\n")

    if new_slots:
        lines.append(
            f"Here are {len(new_slots)} alternative slot(s) with the same doctor"
            f"{preference_suffix}:\n"
        )
        for i, slot in enumerate(new_slots, 1):
            lines.append(
                f"{i}. {slot['label']} "
                f"(doctor_id: {slot['doctor_id']}, "
                f"specialty_id: {slot['specialty_id']}, "
                f"start: {slot['start_at']}, end: {slot['end_at']})"
            )
    else:
        lines.append(
            f"No available slots found with the same doctor{preference_suffix}. "
            f"You can use find_slots with specialty_id {appt['specialty_id']} "
            "to search other doctors."
        )

    return "\n".join(lines)


@tool(args_schema=CancelAppointmentInput)
def cancel_appointment(patient_id: str, appointment_id: str) -> str:
    """Cancel an existing appointment.

    Use this only AFTER the patient has confirmed they want to cancel.
    The appointment status is set to 'cancelled'.
    """
    # Fetch first to confirm it exists and get details for confirmation
    result = (
        supabase.table("appointments")
        .select("id, start_at, status, doctors(full_name), specialties(name)")
        .eq("id", appointment_id)
        .eq("patient_id", patient_id)
        .execute()
    )

    if not result.data:
        return f"Appointment {appointment_id} not found for this patient."

    appt = cast(list[CancelAppointmentRow], result.data)[0]

    if appt["status"] == "cancelled":
        return "This appointment is already cancelled."

    # Cancel it
    try:
        supabase.table("appointments").update({"status": "cancelled"}).eq(
            "id", appointment_id
        ).eq("patient_id", patient_id).execute()
    except Exception:
        logger.exception("Failed to cancel appointment")
        return "Failed to cancel appointment. Please try again."

    label = format_for_voice(appt["start_at"])
    return (
        f"Appointment with {_format_doctor_name_for_voice(appt['doctors']['full_name'])} "
        f"({appt['specialties']['name']}) on {label} has been cancelled."
    )


@tool
def list_specialties() -> str:
    """List all available medical specialties at this clinic.

    Use this when the patient's symptoms don't match any specialty,
    or when they want to browse what's available.
    """
    result = (
        supabase.table("specialties")
        .select("id, name, description")
        .order("name")
        .execute()
    )

    if not result.data:
        return "No specialties found."

    lines = ["Available specialties:\n"]
    for spec in cast(list[SpecialtyListRow], result.data):
        desc = spec.get("description") or "No description"
        lines.append(f"- {spec['name']} (ID: {spec['id']}): {desc}")

    return "\n".join(lines)


# ============================================================
# TOOL REGISTRY
# ============================================================
# Collected here so agent.py can import a single list.
# Order doesn't matter — the LLM picks based on descriptions.
# ============================================================

ALL_TOOLS = [
    find_patient_by_identifier,
    find_patients_by_demographics,
    register_patient,
    triage_symptoms,
    find_slots,
    book_appointment,
    find_appointment,
    reschedule_appointment,
    cancel_appointment,
    list_specialties,
]
