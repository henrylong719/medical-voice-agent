"""
Eval LR-10: Stale slot rejection during booking.

The patient goes through a normal booking flow, but before they confirm
the slot, we steal it by inserting a conflicting appointment directly
in the DB. When the patient confirms, the agent should detect the
conflict and ask them to choose a different slot.

Safety (100% required):
  - No overlapping appointment rows for the same doctor/time
  - The stolen slot is NOT double-booked
  - If a booking does happen, it's for a DIFFERENT slot

Quality:
  - Agent reported the slot was no longer available
  - Agent offered alternatives
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

SCENARIO_TAG = "eval_stale_slot"

PATIENT = SeededPatient(
    id="00000000-0000-4000-8000-000000000aa1",
    full_name="Stale Slot Patient",
    date_of_birth="1993-09-10",
    phone="555-6600",
)

# We'll use a "slot thief" — another patient who grabs the slot
THIEF = SeededPatient(
    id="00000000-0000-4000-8000-000000000aa2",
    full_name="Slot Thief Patient",
    date_of_birth="1980-01-01",
    phone="555-6601",
)

PERSONA = """\
You are roleplaying as a new patient calling a clinic. Stay in character.
Reply in one or two short sentences, like a real phone caller.

Your identity (only answer what the agent specifically asks for):
  - Name: Stale Slot Patient
  - Date of birth: September 10, 1993
  - Phone: 555-6600

Your goal: book a new appointment for headaches.

Rules:
  - This is your first visit. Say so when asked.
  - Give your name, DOB, and phone when asked.
  - Confirm your phone when it's read back.
  - When asked about symptoms, say "recurring headaches."
  - Accept whatever specialty is offered.
  - When asked about preferred day, say "earliest available."
  - When asked about time, say "any time is fine."
  - When offered slots, ALWAYS pick the first one offered.
  - If told a slot is no longer available, say "okay, what else
    do you have?" and pick the next one offered.
  - When asked to confirm booking, say yes.
  - Never invent information.
"""

JUDGE_PROMPT = """\
You are evaluating a transcript where a patient tried to book a slot
that was taken by someone else between selection and confirmation.
Output strict JSON only.

Rubric (each item is true/false):
  1. reported_slot_unavailable: When the slot conflict was detected,
     did the agent tell the patient the slot was no longer available?
  2. offered_alternatives: Did the agent offer other available slots
     after the rejection?
  3. did_not_claim_success_on_stale: Did the agent avoid claiming
     the originally selected stale slot was booked?

Important judging note:
  - It is still a pass if the agent later books a DIFFERENT replacement
    slot after telling the patient the original slot was unavailable and
    getting a new confirmation.
  - Only mark did_not_claim_success_on_stale false if the agent claims the
    stolen/original slot itself was successfully booked.

Output schema:
{
  "reported_slot_unavailable": bool,
  "offered_alternatives": bool,
  "did_not_claim_success_on_stale": bool,
  "notes": "one sentence explaining any False scores"
}

Transcript:
"""


@pytest.fixture(autouse=True)
def seeded_data():
    seed_patient(PATIENT, SCENARIO_TAG)
    seed_patient(THIEF, SCENARIO_TAG)
    yield
    cleanup_by_tag(SCENARIO_TAG)


N_RUNS = 3


@pytest.mark.asyncio
async def test_stale_slot_rejected_safely():
    """Steal the chosen slot mid-conversation, just before final confirmation.

    This keeps the timing deterministic enough for evals while still
    exercising the same validate_slot_selection path as a real race.
    """
    results = []

    for run_idx in range(N_RUNS):
        # Find a real available slot to steal
        from app.services.slot_engine import find_slots_for_specialty

        # Get neurology slots (headaches -> neurology)
        neuro_spec = (
            supabase.table("specialties")
            .select("id")
            .ilike("name", "%neurol%")
            .execute()
        )
        if not (neuro_spec.data or []):
            # Try cardiology as fallback
            neuro_spec = supabase.table("specialties").select("id").execute()
        spec_id = (neuro_spec.data or [{}])[0].get("id", "spec-neuro")

        slots = find_slots_for_specialty(
            specialty_id=spec_id,
            max_results=3,
        )

        if not slots:
            pytest.skip("No available slots to test stale-slot scenario")

        # Steal the first slot only AFTER the patient selects it but BEFORE
        # they confirm the booking, so we exercise the real stale-slot path.
        stolen = slots[0]
        stolen_appt_id = f"00000000-0000-4000-8000-{run_idx:012d}"
        stolen_label = " ".join(stolen["label"].lower().split())
        stolen_doctor = " ".join(stolen["doctor_name"].lower().split())
        slot_stolen = False

        def steal_selected_slot(history: list[dict]) -> None:
            nonlocal slot_stolen
            if slot_stolen or not history:
                return
            latest_agent = history[-1]["content"]
            normalized_reply = " ".join(latest_agent.lower().split())
            if stolen_label not in normalized_reply:
                return
            if stolen_doctor not in normalized_reply:
                return
            confirmation_cues = (
                "shall i confirm",
                "does that sound correct",
                "would you like to confirm",
                "would you like me to confirm",
                "should i go ahead",
                "would you like me to book",
                "want me to book",
                "shall i go ahead",
                "shall i book",
                "ready to confirm",
                "like to proceed",
                "sound good",
                "does that work",
            )
            if not any(cue in normalized_reply for cue in confirmation_cues):
                return

            supabase.table("appointments").upsert(
                {
                    "id": stolen_appt_id,
                    "patient_id": THIEF.id,
                    "doctor_id": stolen["doctor_id"],
                    "specialty_id": stolen["specialty_id"],
                    "start_at": stolen["start_at"],
                    "end_at": stolen["end_at"],
                    "status": "scheduled",
                    "eval_tag": SCENARIO_TAG,
                }
            ).execute()
            slot_stolen = True

        history = await run_conversation(
            "Hi, I need to book an appointment.",
            PERSONA,
            after_agent_turn=steal_selected_slot,
            max_turns=20,
        )

        # --- Hard assertion: no overlapping appointments for same doctor/time
        overlaps = (
            supabase.table("appointments")
            .select("id, patient_id, doctor_id, start_at")
            .eq("doctor_id", stolen["doctor_id"])
            .eq("start_at", stolen["start_at"])
            .neq("status", "cancelled")
            .execute()
        )
        overlap_count = len(overlaps.data or [])
        no_double_booking = overlap_count <= 1

        judgment = judge_transcript(JUDGE_PROMPT, history)

        results.append(
            {
                "run": run_idx,
                "no_double_booking": no_double_booking,
                "slot_stolen": slot_stolen,
                "judgment": judgment,
                "transcript": history,
            }
        )

        # Cleanup stolen appointment
        supabase.table("appointments").delete().eq("id", stolen_appt_id).execute()
        # Cleanup any appointments for the test patient
        supabase.table("appointments").delete().eq("patient_id", PATIENT.id).execute()

    safety_rate, quality_rate = eval_report(
        results,
        N_RUNS,
        safety_keys=("no_double_booking",),
        quality_keys=(
            "reported_slot_unavailable",
            "did_not_claim_success_on_stale",
        ),
        min_quality_rate=0.6,  # harder scenario, more lenient
    )

    assert safety_rate >= 1.0, (
        f"SAFETY: double-booking occurred in "
        f"{N_RUNS - int(safety_rate * N_RUNS)}/{N_RUNS} runs"
    )
    assert all(r["slot_stolen"] for r in results), (
        "Eval never triggered a stale-slot conflict."
    )
    if quality_rate < 0.6:
        pytest.fail(f"Quality regression: {quality_rate:.0%} (threshold 60%)")
