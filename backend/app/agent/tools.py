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

from typing import cast

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.supabase_client import supabase
from app.services.slot_engine import find_slots_for_specialty, find_slots_for_doctor
from app.services.time_utils import format_for_voice
from app.models.db_rows import (
    CancelAppointmentRow,
    CreatedAppointmentRow,
    FindAppointmentRow,
    PatientLookupRow,
    RescheduleAppointmentRow,
    SpecialtyListRow,
    TriageMatchRow,
    TriageScore,
)


# ============================================================
# INPUT SCHEMAS
# ============================================================
# These Pydantic models define what the LLM must provide when
# calling each tool. The Field descriptions are critical — the
# LLM reads them to understand what values to pass.
# ============================================================

class IdentifyPatientInput(BaseModel):
    """Input for looking up a patient."""
    uin: str = Field(description="The patient's 9-digit university ID number (UIN)")


class RegisterPatientInput(BaseModel):
    """Input for registering a new patient."""
    uin: str = Field(description="9-digit university ID number")
    full_name: str = Field(description="Patient's full name")
    phone: str | None = Field(default=None, description="Phone number (optional)")


class TriageSymptomsInput(BaseModel):
    """Input for symptom-based triage."""
    symptoms: list[str] = Field(
        description=(
            "List of individual symptoms the patient described. "
            "Break compound descriptions into separate symptoms. "
            "Example: ['chest pain', 'shortness of breath', 'dizziness']"
        )
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


class RescheduleInput(BaseModel):
    """Input for rescheduling an appointment."""
    appointment_id: str = Field(description="UUID of the appointment to reschedule")
    preferred_day: str | None = Field(
        default=None,
        description="When the patient wants the new appointment (natural language)",
    )
    preferred_time: str | None = Field(
        default=None,
        description="Time of day preference: 'morning', 'afternoon', or empty for any",
    )


class CancelAppointmentInput(BaseModel):
    """Input for cancelling an appointment."""
    appointment_id: str = Field(description="UUID of the appointment to cancel")


# ============================================================
# TOOLS
# ============================================================
# Each @tool function is what the LLM can call. The docstring
# is the tool description the LLM reads. Return strings —
# the LLM needs text it can reason about, not raw dicts.
# ============================================================

@tool(args_schema=IdentifyPatientInput)
def identify_patient(uin: str) -> str:
    """Look up a patient by their 9-digit UIN (university ID number).

    Use this FIRST in every conversation to identify who the patient is.
    Returns the patient's name, ID, and contact info if found.
    If the patient is not found, they may need to register as a new patient.
    """
    result = (
        supabase.table("patients")
        .select("id, uin, full_name, phone, email, allergies")
        .eq("uin", uin)
        .execute()
    )

    if not result.data:
        return f"No patient found with UIN {uin}. They may need to register as a new patient."

    patient = cast(list[PatientLookupRow], result.data)[0]
    allergies = ", ".join(patient.get("allergies") or []) or "None listed"

    return (
        f"Patient found: {patient['full_name']} "
        f"(ID: {patient['id']}, UIN: {patient['uin']}). "
        f"Phone: {patient.get('phone') or 'not on file'}. "
        f"Allergies: {allergies}."
    )


@tool(args_schema=RegisterPatientInput)
def register_patient(uin: str, full_name: str, phone: str | None = None) -> str:
    """Register a new patient who doesn't exist in the system yet.

    Use this only AFTER identify_patient confirms the patient is not found.
    Requires a 9-digit UIN and the patient's full name.
    """
    # Check if UIN already exists
    existing = (
        supabase.table("patients")
        .select("id")
        .eq("uin", uin)
        .execute()
    )
    if existing.data:
        return f"A patient with UIN {uin} already exists. Use identify_patient to look them up."

    data: dict[str, str] = {"uin": uin, "full_name": full_name}
    if phone:
        data["phone"] = phone

    result = supabase.table("patients").insert(data).execute()

    if not result.data:
        return "Failed to register patient. Please try again."

    patient = cast(list[PatientLookupRow], result.data)[0]
    return (
        f"Successfully registered {patient['full_name']} "
        f"(ID: {patient['id']}, UIN: {patient['uin']})."
    )


@tool(args_schema=TriageSymptomsInput)
def triage_symptoms(symptoms: list[str]) -> str:
    """Match patient symptoms to medical specialties using keyword search.

    Takes a list of symptoms and finds matching specialties ranked by relevance.
    Also returns follow-up questions to ask the patient for better assessment.
    Use this after identifying the patient and collecting their symptoms.
    """
    if not symptoms:
        return "No symptoms provided. Please ask the patient to describe their symptoms."

    # Query symptom_specialty_map for each symptom using keyword matching.
    # We use ilike for case-insensitive partial matching.
    # In Phase 3, this gets replaced with semantic (RAG) search.
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

    if not all_matches:
        return (
            f"No specialty matches found for symptoms: {', '.join(symptoms)}. "
            "Consider asking the patient to describe their symptoms differently, "
            "or use list_specialties to show available options."
        )

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

        # Collect unique follow-up questions
        for q in (match.get("follow_up_questions") or []):
            if q not in entry["follow_up_questions"]:
                entry["follow_up_questions"].append(q)

    # Sort by total weight (highest relevance first)
    ranked = sorted(
        specialty_scores.items(),
        key=lambda item: item[1]["total_weight"],
        reverse=True,
    )

    # Format results for the LLM to reason about
    lines = ["Triage results (ranked by relevance):\n"]

    for spec_id, info in ranked:
        matched = ", ".join(info["matched_symptoms"])
        lines.append(
            f"- {info['name']} (ID: {spec_id}): "
            f"score {info['total_weight']:.2f}, "
            f"matched on: {matched}"
        )
        if info["follow_up_questions"]:
            # Include top 2 follow-up questions for the best match
            top_questions = info["follow_up_questions"][:2]
            lines.append(f"  Follow-up questions: {'; '.join(top_questions)}")

    return "\n".join(lines)


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
    Returns up to 5 slots with doctor name, date, and time.
    """
    slots = find_slots_for_specialty(
        specialty_id=specialty_id,
        preferred_day=preferred_day,
        preferred_time=preferred_time,
        max_results=5,
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
            f"{i}. Dr. {slot['doctor_name']} — {slot['label']} "
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

    result = supabase.table("appointments").insert(data).execute()

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
def find_appointment(patient_id: str, doctor_name: str | None = None) -> str:
    """Look up a patient's existing appointments.

    Returns upcoming scheduled appointments. Optionally filter by doctor name.
    Use this when a patient wants to reschedule or cancel an existing appointment.
    """
    query = (
        supabase.table("appointments")
        .select("id, start_at, end_at, status, reason, doctors(full_name), specialties(name)")
        .eq("patient_id", patient_id)
        .eq("status", "scheduled")
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
    preferred_day: str | None = None,
    preferred_time: str | None = None,
) -> str:
    """Reschedule an existing appointment by cancelling it and finding new slots.

    This cancels the old appointment and searches for new available slots
    with the same doctor and specialty. The patient still needs to confirm
    and book a new slot after reviewing options.
    """
    # Fetch the existing appointment
    result = (
        supabase.table("appointments")
        .select("id, doctor_id, specialty_id, start_at, status, doctors(full_name), specialties(name)")
        .eq("id", appointment_id)
        .execute()
    )

    if not result.data:
        return f"Appointment {appointment_id} not found."

    appt = cast(list[RescheduleAppointmentRow], result.data)[0]

    if appt["status"] == "cancelled":
        return "This appointment is already cancelled."

    # Cancel the old appointment
    supabase.table("appointments").update(
        {"status": "cancelled"}
    ).eq("id", appointment_id).execute()

    old_label = format_for_voice(appt["start_at"])

    # Find new slots with the same doctor and specialty
    new_slots = find_slots_for_doctor(
        doctor_id=appt["doctor_id"],
        specialty_id=appt["specialty_id"],
        preferred_day=preferred_day,
        preferred_time=preferred_time,
        max_results=5,
    )

    lines = [
        f"Cancelled appointment with Dr. {appt['doctors']['full_name']} "
        f"({appt['specialties']['name']}) on {old_label}.\n"
    ]

    if new_slots:
        lines.append(f"Here are {len(new_slots)} new slot(s) with the same doctor:\n")
        for i, slot in enumerate(new_slots, 1):
            lines.append(
                f"{i}. {slot['label']} "
                f"(doctor_id: {slot['doctor_id']}, "
                f"start: {slot['start_at']}, end: {slot['end_at']})"
            )
    else:
        lines.append(
            "No available slots found with the same doctor. "
            "You can use find_slots with the specialty_id to search other doctors."
        )

    return "\n".join(lines)


@tool(args_schema=CancelAppointmentInput)
def cancel_appointment(appointment_id: str) -> str:
    """Cancel an existing appointment.

    Use this only AFTER the patient has confirmed they want to cancel.
    The appointment status is set to 'cancelled'.
    """
    # Fetch first to confirm it exists and get details for confirmation
    result = (
        supabase.table("appointments")
        .select("id, start_at, status, doctors(full_name), specialties(name)")
        .eq("id", appointment_id)
        .execute()
    )

    if not result.data:
        return f"Appointment {appointment_id} not found."

    appt = cast(list[CancelAppointmentRow], result.data)[0]

    if appt["status"] == "cancelled":
        return "This appointment is already cancelled."

    # Cancel it
    supabase.table("appointments").update(
        {"status": "cancelled"}
    ).eq("id", appointment_id).execute()

    label = format_for_voice(appt["start_at"])
    return (
        f"Appointment with Dr. {appt['doctors']['full_name']} "
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
    identify_patient,
    register_patient,
    triage_symptoms,
    find_slots,
    book_appointment,
    find_appointment,
    reschedule_appointment,
    cancel_appointment,
    list_specialties,
]
