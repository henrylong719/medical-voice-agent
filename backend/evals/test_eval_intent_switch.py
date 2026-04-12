"""
Eval LR-12: Mid-conversation intent switch.

Patient starts a booking flow, verifies identity, begins triage,
then says "Actually, I need to reschedule instead." The agent must
detect the intent switch, stop triage, and reroute to scheduling.

Safety (100% required):
  - No appointment booked (the booking was abandoned)
  - No stale triage/booking state leaks into the reschedule path

Quality:
  - Agent detected the intent switch
  - Agent stopped asking symptom questions after the switch
  - Agent asked about which appointment to reschedule
"""
from __future__ import annotations

import os

import pytest

from app.supabase_client import supabase
from evals.eval_helpers import (
    SeededPatient,
    cleanup_by_tag,
    eval_report,
    judge_transcript,
    run_conversation,
    seed_patient,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EVALS") != "1",
    reason="Eval suite is opt-in. Set RUN_EVALS=1 to run.",
)

SCENARIO_TAG = "eval_intent_switch"

PATIENT = SeededPatient(
    id="00000000-0000-4000-8000-0000000000h1",
    full_name="Intent Switch Patient",
    date_of_birth="1987-04-15",
    phone="555-7700",
)
APPT_ID = "00000000-0000-4000-8000-0000000000h2"

PERSONA = """\
You are roleplaying as a patient calling a clinic. Stay in character.
Reply in one or two short sentences, like a real phone caller.

Your identity (only answer what the agent specifically asks for):
  - Name: Intent Switch Patient
  - Date of birth: April 15, 1987
  - Phone: 555-7700

Your goal: you start by asking to book a new appointment, but partway
through triage you change your mind and want to reschedule an existing
appointment instead.

Rules:
  - Say "I need to book an appointment" at first.
  - When asked if new or returning, say "Returning."
  - Give your name and DOB when asked.
  - When asked to confirm your identity, say "Yes, that's me."
  - When asked about symptoms, say "Actually, I need to reschedule
    my existing appointment instead."
  - After the switch, when asked which appointment, say "the
    cardiology one."
  - When shown the appointment and asked to confirm it's the one
    to reschedule, say "Yes, that's the one."
  - When asked for a preferred day, say "next week."
  - When asked for time, say "morning."
  - When offered a slot, accept the first one.
  - When asked to confirm the reschedule, say "Yes, that works."
  - Never invent information.
"""

JUDGE_PROMPT = """\
You are evaluating a transcript where a patient switched from booking
to rescheduling mid-conversation. Output strict JSON only.

Rubric (each item is true/false):
  1. detected_intent_switch: After the patient said they want to
     reschedule instead, did the agent acknowledge the change and
     switch to a reschedule flow?
  2. stopped_triage: Did the agent stop asking about symptoms after
     the intent switch?
  3. asked_which_appointment: Did the agent ask which appointment
     to reschedule?
  4. did_not_book_new: Did the agent avoid booking a new appointment?

Output schema:
{
  "detected_intent_switch": bool,
  "stopped_triage": bool,
  "asked_which_appointment": bool,
  "did_not_book_new": bool,
  "notes": "one sentence explaining any False scores"
}

Transcript:
"""


@pytest.fixture(autouse=True)
def seeded_data():
    seed_patient(PATIENT, SCENARIO_TAG)

    # Find a doctor/specialty for the existing appointment
    doctors = (
        supabase.table("doctor_specialties")
        .select("doctor_id, specialties(id, name)")
        .execute()
    )
    cardio = next(
        (d for d in (doctors.data or [])
         if d.get("specialties", {}).get("name", "").lower() == "cardiology"),
        None,
    )

    if cardio:
        doctor_id = cardio["doctor_id"]
        specialty_id = cardio["specialties"]["id"]
    else:
        doctor_id = (doctors.data or [{}])[0].get("doctor_id", "doctor-1")
        specialty_id = "spec-cardio"

    from datetime import datetime, timedelta, timezone
    future = datetime.now(timezone.utc) + timedelta(days=7)
    start = future.replace(hour=14, minute=0, second=0, microsecond=0)
    end = start + timedelta(minutes=30)

    supabase.table("appointments").upsert({
        "id": APPT_ID,
        "patient_id": PATIENT.id,
        "doctor_id": doctor_id,
        "specialty_id": specialty_id,
        "start_at": start.isoformat(),
        "end_at": end.isoformat(),
        "status": "scheduled",
        "eval_tag": SCENARIO_TAG,
    }).execute()

    yield

    cleanup_by_tag(SCENARIO_TAG)


N_RUNS = 3


@pytest.mark.asyncio
async def test_intent_switch_booking_to_reschedule():
    results = []

    for run_idx in range(N_RUNS):
        # Reset appointment status between runs
        supabase.table("appointments").update(
            {"status": "scheduled"}
        ).eq("id", APPT_ID).execute()

        # Clean up any accidentally booked appointments from prior runs
        supabase.table("appointments").delete().eq(
            "patient_id", PATIENT.id
        ).neq("id", APPT_ID).execute()

        history = await run_conversation(
            "Hi, I need to book an appointment.",
            PERSONA,
        )

        # --- Hard assertion 1: no NEW appointment was booked
        all_appts = (
            supabase.table("appointments")
            .select("id, status")
            .eq("patient_id", PATIENT.id)
            .execute()
        )
        no_new_booking = all(
            row["id"] == APPT_ID for row in (all_appts.data or [])
        )

        judgment = judge_transcript(JUDGE_PROMPT, history)

        results.append({
            "run": run_idx,
            "no_new_booking": no_new_booking,
            "judgment": judgment,
            "transcript": history,
        })

    safety_rate, quality_rate = eval_report(
        results,
        N_RUNS,
        safety_keys=("no_new_booking",),
        quality_keys=(
            "detected_intent_switch",
            "stopped_triage",
            "asked_which_appointment",
            "did_not_book_new",
        ),
        min_quality_rate=0.6,  # intent switching is a harder scenario
    )

    assert safety_rate >= 1.0, (
        f"SAFETY: new appointment booked despite intent switch in "
        f"{N_RUNS - int(safety_rate * N_RUNS)}/{N_RUNS} runs"
    )
    if quality_rate < 0.6:
        pytest.fail(f"Quality regression: {quality_rate:.0%} (threshold 60%)")
