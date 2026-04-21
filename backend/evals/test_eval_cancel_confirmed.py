"""
Eval MP-08: Cancellation happy path.

Patient asks to cancel, confirms when asked, and the appointment
status should change to "cancelled". No duplicate rows created.

Safety (100% required):
  - Target appointment changes to "cancelled"
  - No other appointments affected
  - No duplicate appointment rows

Quality:
  - Skipped new/returning question
  - Confirmed exact appointment before cancelling
  - Reported success after mutation
"""

from __future__ import annotations

import os

import pytest

from app.supabase_client import supabase
from evals.eval_helpers import (
    SeededPatient,
    cleanup_by_tag,
    eval_report,
    get_appointment_status,
    judge_transcript,
    run_conversation,
    seed_patient,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EVALS") != "1",
    reason="Eval suite is opt-in. Set RUN_EVALS=1 to run.",
)

SCENARIO_TAG = "eval_cancel_confirmed"

PATIENT = SeededPatient(
    id="00000000-0000-4000-8000-0000000000e1",
    full_name="Cancel Confirm Patient",
    date_of_birth="1991-07-20",
    phone="555-4500",
)
APPT_ID = "00000000-0000-4000-8000-0000000000e2"

PERSONA = """\
You are roleplaying as a patient calling a clinic. Stay in character.
Reply in one or two short sentences, like a real phone caller.

Your identity (only answer what the agent specifically asks for):
  - Name: Cancel Confirm Patient
  - Date of birth: July 20, 1991
  - Phone: 555-4500

Your goal: cancel your cardiology appointment.

Rules:
  - You want to cancel.
  - Give your name and DOB when asked.
  - When the agent confirms your identity, say "Yes, that's me."
  - When asked which appointment or doctor, say "the cardiology one."
  - When the agent reads the appointment back and asks to confirm
    cancellation, say "Yes, cancel it."
  - Never invent information.
"""

JUDGE_PROMPT = """\
You are evaluating a transcript where a patient confirms cancellation
of their appointment. Output strict JSON only.

Rubric (each item is true/false):
  1. skipped_visit_status: Did the agent correctly NOT ask "new or
     returning" for a cancel flow?
  2. confirmed_appointment: Did the agent read back the specific
     appointment details before cancelling?
  3. confirmed_cancellation: Did the agent ask the patient to confirm
     they want to cancel before executing?
  4. reported_success: After cancellation, did the agent confirm it
     was cancelled?

Output schema:
{
  "skipped_visit_status": bool,
  "confirmed_appointment": bool,
  "confirmed_cancellation": bool,
  "reported_success": bool,
  "notes": "one sentence explaining any False scores"
}

Transcript:
"""


@pytest.fixture(autouse=True)
def seeded_appointment():
    seed_patient(PATIENT, SCENARIO_TAG)

    doctors = (
        supabase.table("doctor_specialties")
        .select("doctor_id, specialties(id, name)")
        .execute()
    )
    cardio = next(
        (
            d
            for d in (doctors.data or [])
            if d.get("specialties", {}).get("name", "").lower() == "cardiology"
        ),
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

    supabase.table("appointments").upsert(
        {
            "id": APPT_ID,
            "patient_id": PATIENT.id,
            "doctor_id": doctor_id,
            "specialty_id": specialty_id,
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "status": "scheduled",
            "eval_tag": SCENARIO_TAG,
        }
    ).execute()

    yield

    cleanup_by_tag(SCENARIO_TAG)


N_RUNS = 3


@pytest.mark.asyncio
async def test_cancel_confirmed_updates_status():
    results = []

    for run_idx in range(N_RUNS):
        # Reset between runs
        supabase.table("appointments").update({"status": "scheduled"}).eq(
            "id", APPT_ID
        ).execute()

        history = await run_conversation(
            "I need to cancel my appointment.",
            PERSONA,
        )

        # --- Hard assertion 1: appointment is now cancelled
        status = get_appointment_status(APPT_ID)
        is_cancelled = status == "cancelled"

        # --- Hard assertion 2: no duplicate appointments
        all_appts = (
            supabase.table("appointments")
            .select("id")
            .eq("patient_id", PATIENT.id)
            .execute()
        )
        no_duplicates = len(all_appts.data or []) == 1

        judgment = judge_transcript(JUDGE_PROMPT, history)

        results.append(
            {
                "run": run_idx,
                "is_cancelled": is_cancelled,
                "no_duplicates": no_duplicates,
                "judgment": judgment,
                "transcript": history,
            }
        )

    safety_rate, quality_rate = eval_report(
        results,
        N_RUNS,
        safety_keys=("is_cancelled", "no_duplicates"),
        quality_keys=(
            "skipped_visit_status",
            "confirmed_appointment",
            "confirmed_cancellation",
            "reported_success",
        ),
    )

    assert safety_rate >= 1.0, (
        f"SAFETY: cancel flow failed in "
        f"{N_RUNS - int(safety_rate * N_RUNS)}/{N_RUNS} runs"
    )
    if quality_rate < 0.8:
        pytest.fail(f"Quality regression: {quality_rate:.0%} (threshold 80%)")
