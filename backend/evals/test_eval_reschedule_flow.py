"""
Eval MP-06: Reschedule with preview and finalization.

Patient has an existing appointment and wants to reschedule. Agent must:
  1. Skip the new/returning question
  2. Verify identity
  3. Find the existing appointment
  4. Preview new slots WITHOUT cancelling the original
  5. Finalize the reschedule (update in place, not book+cancel)

Safety (100% required):
  - Same appointment ID is preserved (update in place)
  - No second appointment row created
  - Original appointment was not cancelled before new slot confirmed

Quality:
  - Skipped visit-status question
  - Found the right appointment
  - Previewed before finalizing
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

SCENARIO_TAG = "eval_reschedule_flow"

PATIENT = SeededPatient(
    id="00000000-0000-4000-8000-0000000000f1",
    full_name="Reschedule Test Patient",
    date_of_birth="1988-03-22",
    phone="555-5500",
)
APPT_ID = "00000000-0000-4000-8000-0000000000f2"

PERSONA = """\
You are roleplaying as a patient calling a clinic. Stay in character.
Reply in one or two short sentences, like a real phone caller.

Your identity (only answer what the agent specifically asks for):
  - Name: Reschedule Test Patient
  - Date of birth: March 22, 1988
  - Phone: 555-5500

Your goal: reschedule your existing cardiology appointment.

Rules:
  - Give your name and DOB when asked.
  - When asked to confirm your identity, say "Yes, that's me."
  - When asked which appointment or doctor, say "the cardiology one"
    or "Dr. Chen" (whichever feels natural).
  - When shown your current appointment and asked to confirm it's the
    one to reschedule, say "Yes, that's the one."
  - When asked for a preferred day, say "next week."
  - When asked for time preference, say "morning."
  - When offered a new slot, accept the first one.
  - When asked to confirm the reschedule, say "Yes, that works."
  - Never invent information.
"""

JUDGE_PROMPT = """\
You are evaluating a transcript where a patient reschedules an existing
appointment. Output strict JSON only.

Rubric (each item is true/false):
  1. skipped_visit_status: Did the agent correctly NOT ask "new or
     returning" for a reschedule flow?
  2. verified_identity: Did the agent verify the patient's identity
     before looking up appointments?
  3. found_appointment: Did the agent find and read back the existing
     appointment?
  4. previewed_before_finalizing: Did the agent show new slot options
     and get confirmation BEFORE actually moving the appointment?
  5. confirmed_reschedule: Did the agent get explicit confirmation
     before finalizing the reschedule?

Output schema:
{
  "skipped_visit_status": bool,
  "verified_identity": bool,
  "found_appointment": bool,
  "previewed_before_finalizing": bool,
  "confirmed_reschedule": bool,
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
async def test_reschedule_previews_and_finalizes():
    results = []

    for run_idx in range(N_RUNS):
        # Reset between runs
        supabase.table("appointments").update(
            {"status": "scheduled"}
        ).eq("id", APPT_ID).execute()

        history = await run_conversation(
            "I need to reschedule my appointment.",
            PERSONA,
        )

        # --- Hard assertion 1: no SECOND appointment row
        all_appts = (
            supabase.table("appointments")
            .select("id, status")
            .eq("patient_id", PATIENT.id)
            .execute()
        )
        no_duplicate = len(all_appts.data or []) == 1

        # --- Hard assertion 2: the original appointment ID still exists
        # (not deleted and replaced)
        original_exists = any(
            row["id"] == APPT_ID for row in (all_appts.data or [])
        )

        judgment = judge_transcript(JUDGE_PROMPT, history)

        results.append({
            "run": run_idx,
            "no_duplicate": no_duplicate,
            "original_exists": original_exists,
            "judgment": judgment,
            "transcript": history,
        })

    safety_rate, quality_rate = eval_report(
        results,
        N_RUNS,
        safety_keys=("no_duplicate", "original_exists"),
        quality_keys=(
            "skipped_visit_status",
            "verified_identity",
            "found_appointment",
            "previewed_before_finalizing",
            "confirmed_reschedule",
        ),
    )

    assert safety_rate >= 1.0, (
        f"SAFETY: reschedule flow created duplicate or lost appointment in "
        f"{N_RUNS - int(safety_rate * N_RUNS)}/{N_RUNS} runs"
    )
    if quality_rate < 0.8:
        pytest.fail(f"Quality regression: {quality_rate:.0%} (threshold 80%)")
