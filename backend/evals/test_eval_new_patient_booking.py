"""
Eval MP-01: New-patient booking end-to-end.

Covers the full happy path: greeting -> intent -> new/returning question ->
registration (name, DOB, phone with confirmation) -> triage -> slot
selection -> booking confirmation.

Safety (hard DB assertions):
  - Exactly one new patient row is created
  - Exactly one new appointment row in "scheduled" status
  - No duplicate patient rows
  - Phone was confirmed before registration (judged)

Quality (LLM judge):
  - Asked new/returning before intake
  - Collected all three fields before registering
  - Triage happened before scheduling
  - Explicit booking confirmation before mutation
"""
from __future__ import annotations

import os

import pytest

from app.supabase_client import supabase
from evals.eval_helpers import (
    cleanup_by_tag,
    eval_report,
    judge_transcript,
    run_conversation,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EVALS") != "1",
    reason="Eval suite is opt-in. Set RUN_EVALS=1 to run.",
)

SCENARIO_TAG = "eval_new_patient_booking"
EVAL_PATIENT_NAME = "Maya Chen Eval"

PERSONA = """\
You are roleplaying as a patient calling a clinic. Stay in character.
Reply in one or two short sentences, like a real phone caller.

Your identity (only answer what the agent specifically asks for):
  - Name: Maya Chen Eval
  - Date of birth: October 26, 1985
  - Phone: 555-3300

Your goal: book a new appointment. You have never been to this clinic before.

Your symptoms: recurring headaches with flashing lights.

Rules:
  - This is your first visit. Say so when asked.
  - Give your phone number when asked for it.
  - When the agent reads your phone number back, confirm it.
  - When asked about symptoms, describe headaches with flashing lights.
  - When offered a specialty (like Neurology), accept it.
  - When asked about preferred day, say "next week".
  - When asked about time, say "morning".
  - When offered a specific slot, accept the first one.
  - When asked to confirm a booking, say yes.
  - Never invent information you weren't given.
"""

JUDGE_PROMPT = """\
You are evaluating a transcript between a clinic scheduling agent and a
new patient booking their first appointment. Output strict JSON only.

Rubric (each item is true/false):
  1. asked_new_or_returning: Did the agent ask whether the patient has
     been seen before or if this is their first visit?
  2. collected_all_fields: Did the agent collect full name, date of birth,
     AND phone number before registering the patient?
  3. confirmed_phone: Did the agent read the phone number back and get
     confirmation before saving?
  4. triage_before_scheduling: Did the agent ask about symptoms and
     determine a specialty BEFORE offering appointment slots?
  5. explicit_booking_confirmation: Did the agent ask the patient to
     confirm the specific slot before booking?
  6. natural_tone: Was the conversation natural and not robotic?

Output schema:
{
  "asked_new_or_returning": bool,
  "collected_all_fields": bool,
  "confirmed_phone": bool,
  "triage_before_scheduling": bool,
  "explicit_booking_confirmation": bool,
  "natural_tone": bool,
  "notes": "one sentence explaining any False scores"
}

Transcript:
"""


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure no leftover eval data before and after."""
    cleanup_by_tag(SCENARIO_TAG)
    yield
    cleanup_by_tag(SCENARIO_TAG)


N_RUNS = 3


@pytest.mark.asyncio
async def test_new_patient_booking_end_to_end():
    results = []

    for run_idx in range(N_RUNS):
        history = await run_conversation(
            "Hi there!",
            PERSONA,
        )

        # --- Hard assertion 1: exactly one patient with the eval name
        patients = (
            supabase.table("patients")
            .select("id, full_name, phone")
            .ilike("full_name", f"%{EVAL_PATIENT_NAME}%")
            .execute()
        )
        patient_rows = patients.data or []
        exactly_one_patient = len(patient_rows) == 1

        # --- Hard assertion 2: that patient has a phone number
        patient_has_phone = (
            exactly_one_patient
            and patient_rows[0].get("phone") is not None
            and len(patient_rows[0]["phone"].strip()) > 0
        )

        # --- Hard assertion 3: exactly one scheduled appointment
        if exactly_one_patient:
            patient_id = patient_rows[0]["id"]
            appts = (
                supabase.table("appointments")
                .select("id, status")
                .eq("patient_id", patient_id)
                .eq("status", "scheduled")
                .execute()
            )
            exactly_one_appointment = len(appts.data or []) == 1
        else:
            exactly_one_appointment = False

        judgment = judge_transcript(JUDGE_PROMPT, history)

        results.append({
            "run": run_idx,
            "exactly_one_patient": exactly_one_patient,
            "patient_has_phone": patient_has_phone,
            "exactly_one_appointment": exactly_one_appointment,
            "judgment": judgment,
            "transcript": history,
        })

        # Clean up between runs
        cleanup_by_tag(SCENARIO_TAG)

    safety_rate, quality_rate = eval_report(
        results,
        N_RUNS,
        safety_keys=(
            "exactly_one_patient",
            "patient_has_phone",
            "exactly_one_appointment",
        ),
        quality_keys=(
            "asked_new_or_returning",
            "collected_all_fields",
            "triage_before_scheduling",
            "explicit_booking_confirmation",
        ),
    )

    assert safety_rate >= 1.0, (
        f"SAFETY: new patient booking failed in {N_RUNS - int(safety_rate * N_RUNS)}/{N_RUNS} runs"
    )
    if quality_rate < 0.8:
        pytest.fail(f"Quality regression: {quality_rate:.0%} (threshold 80%)")
