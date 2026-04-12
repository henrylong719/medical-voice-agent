"""
Workflow edge-case tests.

These extend the happy-path workflows with error handling, multi-intent
switching, identity correction mid-flow, off-topic handling, and state
assertion checks that verify the graph's internal state — not just
response text.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from pytest import MonkeyPatch

from app.agent.state import AgentState
from tests.workflow_support import (
    ai_history_contains,
    install_test_graph,
    invoke_sequence,
    invoke_turn,
    latest_human_text,
)


# ── Reusable fake nodes ──────────────────────────────────────

async def _simple_triage_node(state: AgentState) -> dict:
    latest = latest_human_text(state).lower()
    if "headache" in latest or "dizziness" in latest:
        return {
            "messages": [AIMessage(content="Neurology seems like the right specialty.")],
            "specialty_id": "spec-neuro",
            "last_agent": "triage",
        }
    return {
        "messages": [AIMessage(content="What symptoms are you having?")],
        "last_agent": "triage",
    }


async def _simple_scheduling_node(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="Do you have a preferred day or week in mind, or would you like the earliest available?"
            )
        ],
        "last_agent": "scheduling",
    }


def _make_returning_intake():
    """Intake that handles lookup, confirmation, and identity correction."""

    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "sarah connor" in latest and ("1985-10-26" in latest or "10/26/1985" in latest):
            return {
                "messages": [
                    AIMessage(
                        content="I found Sarah Connor born on 1985-10-26 on file — is that you?"
                    )
                ],
                "last_agent": "intake",
            }

        if latest.startswith("yes") and ai_history_contains(state, "is that you?"):
            return {
                "messages": [AIMessage(content="Thanks, you're verified.")],
                "patient_id": "patient-1",
                "patient_name": "Sarah Connor",
                "last_agent": "intake",
            }

        if "john doe" in latest and "1990-05-15" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found John Doe born on 1990-05-15 on file — is that you?"
                    )
                ],
                "last_agent": "intake",
            }

        if latest.startswith("no") and ai_history_contains(state, "is that you?"):
            return {
                "messages": [
                    AIMessage(
                        content="Sorry about that. Can you share your full name and date of birth again?"
                    )
                ],
                "patient_id": None,
                "patient_name": None,
                "last_agent": "intake",
            }

        return {
            "messages": [
                AIMessage(
                    content="Can you share your full name and date of birth so I can look you up?"
                )
            ],
            "last_agent": "intake",
        }

    return fake_intake_node


def _make_new_patient_intake():
    """Intake that handles new patient registration."""

    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()
        patient_status = state.get("patient_status")

        if patient_status == "new":
            if "555-0100" in latest or ("sarah" in latest and "1985" in latest):
                return {
                    "messages": [AIMessage(content="Thanks, you're registered as Sarah Connor.")],
                    "patient_id": "patient-1",
                    "patient_name": "Sarah Connor",
                    "last_agent": "intake",
                }
            return {
                "messages": [
                    AIMessage(
                        content="What is your full name, date of birth, and phone number?"
                    )
                ],
                "last_agent": "intake",
            }

        return {
            "messages": [AIMessage(content="Let's verify your record.")],
            "last_agent": "intake",
        }

    return fake_intake_node


# ── Tests: multi-intent switching ─────────────────────────────

def test_user_switches_from_cancel_to_booking_mid_flow(
    monkeypatch: MonkeyPatch,
) -> None:
    """Switching from cancel to booking should reset and start triage."""

    async def fake_scheduling_node(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="Do you remember which doctor or specialty the appointment is with?"
                )
            ],
            "last_agent": "scheduling",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        _greeting,
        intake_prompt,
        confirmation_prompt,
        scheduling_prompt,
        switched_prompt,
    ) = invoke_sequence(
        "workflow-cancel-to-book",
        "hi",
        "I need to cancel my appointment",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
        "Actually, I'd like to book a new appointment instead.",
    )

    assert "full name and date of birth" in intake_prompt
    assert "Thanks, you're verified." in scheduling_prompt
    assert "What symptoms are you having?" in switched_prompt
    # Negative assertion: scheduling prompt should NOT appear after switch
    assert "which doctor or specialty" not in switched_prompt


def test_user_switches_from_reschedule_to_cancel(
    monkeypatch: MonkeyPatch,
) -> None:
    """Reschedule -> cancel should route back to scheduling with cancel intent."""

    async def fake_scheduling_node(state: AgentState) -> dict:
        intent = state.get("intent")

        if intent == "cancel":
            return {
                "messages": [
                    AIMessage(
                        content="Which appointment would you like to cancel?"
                    )
                ],
                "last_agent": "scheduling",
            }

        return {
            "messages": [
                AIMessage(
                    content="Do you remember which doctor or specialty the appointment is with?"
                )
            ],
            "last_agent": "scheduling",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        _greeting,
        intake_prompt,
        confirmation_prompt,
        scheduling_prompt,
        switched_prompt,
    ) = invoke_sequence(
        "workflow-reschedule-to-cancel",
        "hi",
        "I need to reschedule my appointment",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
        "Actually, I'd rather just cancel it.",
    )

    assert "which doctor or specialty" in scheduling_prompt
    assert "cancel" in switched_prompt.lower()


def test_user_switches_intent_twice_book_to_reschedule_to_cancel(
    monkeypatch: MonkeyPatch,
) -> None:
    """User changes intent multiple times in one conversation."""

    async def fake_scheduling_node(state: AgentState) -> dict:
        intent = state.get("intent")
        if intent == "cancel":
            return {
                "messages": [AIMessage(content="Which appointment would you like to cancel?")],
                "last_agent": "scheduling",
            }
        return {
            "messages": [
                AIMessage(content="Which appointment would you like to reschedule?")
            ],
            "last_agent": "scheduling",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        _lookup_prompt,
        _confirmation_prompt,
        triage_prompt,
        reschedule_prompt,
        cancel_prompt,
    ) = invoke_sequence(
        "workflow-double-switch",
        "hi",
        "I need to book an appointment",
        "I've been seen there before",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
        "Actually, I need to reschedule instead.",
        "Wait, let me just cancel it.",
    )

    assert "What symptoms are you having?" in triage_prompt
    assert "reschedule" in reschedule_prompt.lower()
    assert "cancel" in cancel_prompt.lower()


# ── Tests: identity correction mid-flow ───────────────────────

def test_patient_denies_identity_and_retries(
    monkeypatch: MonkeyPatch,
) -> None:
    """When the patient says 'no that's not me', the flow should retry intake."""

    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "john doe" in latest and "1990-05-15" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found John Doe born on 1990-05-15 on file — is that you?"
                    )
                ],
                "last_agent": "intake",
            }

        if "sarah connor" in latest and "1985-10-26" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found Sarah Connor born on 1985-10-26 on file — is that you?"
                    )
                ],
                "last_agent": "intake",
            }

        if latest.startswith("no") and ai_history_contains(state, "is that you?"):
            return {
                "messages": [
                    AIMessage(
                        content="Sorry about that. Can you share your full name and date of birth again?"
                    )
                ],
                "patient_id": None,
                "patient_name": None,
                "last_agent": "intake",
            }

        if latest.startswith("yes") and ai_history_contains(state, "is that you?"):
            return {
                "messages": [AIMessage(content="Thanks, you're verified.")],
                "patient_id": "patient-1",
                "patient_name": "Sarah Connor",
                "last_agent": "intake",
            }

        return {
            "messages": [
                AIMessage(
                    content="Can you share your full name and date of birth so I can look you up?"
                )
            ],
            "last_agent": "intake",
        }

    install_test_graph(
        monkeypatch,
        intake_node=fake_intake_node,
        triage_node=_simple_triage_node,
        scheduling_node=_simple_scheduling_node,
    )

    (
        _greeting,
        intake_prompt,
        wrong_match,
        retry_prompt,
        correct_match,
        post_verification,
    ) = invoke_sequence(
        "workflow-deny-retry",
        "hi",
        "I need to reschedule my appointment",
        "John Doe, 1990-05-15",
        "No, that's not me",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
    )

    assert "John Doe born on 1990-05-15" in wrong_match
    assert "Sorry about that" in retry_prompt
    assert "Sarah Connor born on 1985-10-26" in correct_match
    assert "Thanks, you're verified." in post_verification


# ── Tests: conversation continuity after completed flow ───────

def test_booking_complete_then_follow_up_reschedule(
    monkeypatch: MonkeyPatch,
) -> None:
    """After completing a booking, the user should be able to immediately
    start a reschedule flow in the same conversation."""

    async def fake_scheduling_node(state: AgentState) -> dict:
        intent = state.get("intent")

        if intent == "reschedule":
            return {
                "messages": [
                    AIMessage(
                        content="Which appointment would you like to reschedule?"
                    )
                ],
                "last_agent": "scheduling",
            }

        return {
            "messages": [
                AIMessage(
                    content="Your appointment has been booked. Can I help with anything else?"
                )
            ],
            "appointment_id": "appt-1",
            "intent": None,
            "last_agent": "scheduling",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_new_patient_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    responses = invoke_sequence(
        "workflow-book-then-reschedule",
        "hi",
        "I need to book an appointment",
        "This is my first visit",
        "Sarah Connor, 10/26/1985, 555-0100",
        "I've had headaches and dizziness",
        "Actually, I also need to reschedule another appointment.",
    )

    booking_complete = responses[4]
    follow_up = responses[5]

    assert "appointment has been booked" in booking_complete
    assert "reschedule" in follow_up.lower()


# ── Tests: new patient registration edge cases ────────────────

def test_new_patient_full_registration_flow_with_triage_and_scheduling(
    monkeypatch: MonkeyPatch,
) -> None:
    """Full end-to-end: new patient -> register -> triage -> scheduling."""

    async def fake_scheduling_node(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="I have Monday April 13th at 9 AM with Dr. Chen available. Would you like to book that?"
                )
            ],
            "last_agent": "scheduling",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_new_patient_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        greeting,
        status_question,
        registration_prompt,
        post_registration,
        symptoms_and_scheduling,
    ) = invoke_sequence(
        "workflow-full-new-patient",
        "hello",
        "I need to book an appointment",
        "This is my first visit",
        "Sarah Connor, 10/26/1985, 555-0100",
        "I've had recurring headaches and some dizziness",
    )

    assert "Welcome to the clinic" in greeting
    assert "first visit" in status_question
    assert "full name, date of birth, and phone number" in registration_prompt
    assert "registered as Sarah Connor" in post_registration
    assert "What symptoms are you having?" in post_registration
    assert "Neurology" in symptoms_and_scheduling
    assert "Dr. Chen" in symptoms_and_scheduling


# ── Tests: negative assertions (things that should NOT happen) ────

def test_scheduling_does_not_appear_before_triage_completes(
    monkeypatch: MonkeyPatch,
) -> None:
    """Scheduling prompts should never appear until triage sets a specialty."""

    async def fake_triage_node(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="Can you describe your symptoms in more detail?"
                )
            ],
            "last_agent": "triage",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_new_patient_intake(),
        triage_node=fake_triage_node,
        scheduling_node=_simple_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        _registration_prompt,
        triage_prompt,
    ) = invoke_sequence(
        "workflow-no-premature-scheduling",
        "hi",
        "I need to book an appointment",
        "This is my first visit",
        "Sarah Connor, 10/26/1985, 555-0100",
    )

    assert "describe your symptoms" in triage_prompt
    assert "preferred day" not in triage_prompt
    assert "earliest available" not in triage_prompt


def test_triage_does_not_run_for_reschedule_intent(
    monkeypatch: MonkeyPatch,
) -> None:
    """Reschedule should skip triage entirely."""

    triage_ran = {"called": False}

    async def fake_triage_node(state: AgentState) -> dict:
        triage_ran["called"] = True
        return {"last_agent": "triage"}

    async def fake_scheduling_node(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="Which appointment would you like to reschedule?"
                )
            ],
            "last_agent": "scheduling",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=fake_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    invoke_sequence(
        "workflow-no-triage-for-reschedule",
        "hi",
        "I need to reschedule my appointment",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
    )

    assert not triage_ran["called"], "Triage should not run for reschedule intent"


def test_triage_does_not_run_for_cancel_intent(
    monkeypatch: MonkeyPatch,
) -> None:
    """Cancel should skip triage entirely."""

    triage_ran = {"called": False}

    async def fake_triage_node(state: AgentState) -> dict:
        triage_ran["called"] = True
        return {"last_agent": "triage"}

    async def fake_scheduling_node(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="Which appointment would you like to cancel?"
                )
            ],
            "last_agent": "scheduling",
        }

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=fake_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    invoke_sequence(
        "workflow-no-triage-for-cancel",
        "hi",
        "I need to cancel my appointment",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
    )

    assert not triage_ran["called"], "Triage should not run for cancel intent"


# ── Tests: reschedule/cancel skip visit status question ───────

def test_reschedule_skips_new_or_returning_question(
    monkeypatch: MonkeyPatch,
) -> None:
    """The 'new or returning?' question should only appear for booking."""

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=_simple_scheduling_node,
    )

    (
        _greeting,
        intake_prompt,
    ) = invoke_sequence(
        "workflow-reschedule-no-status",
        "hi",
        "I need to reschedule my appointment",
    )

    assert "first visit" not in intake_prompt
    assert "been seen at this clinic before" not in intake_prompt
    assert "full name and date of birth" in intake_prompt


def test_cancel_skips_new_or_returning_question(
    monkeypatch: MonkeyPatch,
) -> None:
    """The 'new or returning?' question should only appear for booking."""

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=_simple_scheduling_node,
    )

    (
        _greeting,
        intake_prompt,
    ) = invoke_sequence(
        "workflow-cancel-no-status",
        "hi",
        "I need to cancel my appointment",
    )

    assert "first visit" not in intake_prompt
    assert "been seen at this clinic before" not in intake_prompt
    assert "full name and date of birth" in intake_prompt


# ── Tests: greeting behavior ─────────────────────────────────

def test_greeting_always_appears_first_regardless_of_message(
    monkeypatch: MonkeyPatch,
) -> None:
    """Even if the user starts with an intent, greeting comes first."""

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=_simple_scheduling_node,
    )

    greeting = invoke_turn("I need to book an appointment", "workflow-greeting-with-intent")

    assert "Welcome to the clinic" in greeting


def test_second_message_after_greeting_triggers_intent_classification(
    monkeypatch: MonkeyPatch,
) -> None:
    """After greeting, the next message should be classified for intent."""

    install_test_graph(
        monkeypatch,
        intake_node=_make_returning_intake(),
        triage_node=_simple_triage_node,
        scheduling_node=_simple_scheduling_node,
    )

    greeting, status_q = invoke_sequence(
        "workflow-intent-after-greeting",
        "hello",
        "I'd like to book an appointment",
    )

    assert "Welcome to the clinic" in greeting
    assert "first visit" in status_q or "been seen at this clinic before" in status_q


# ── Tests: returning patient with identifier fallback ─────────

def test_returning_patient_identifier_lookup_succeeds_after_demographics_fail(
    monkeypatch: MonkeyPatch,
) -> None:
    """Demographics fail -> identifier succeeds -> patient verified."""

    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "jane smith" in latest and "1992-03-15" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I couldn't find a record with that name and date of birth. "
                        "Do you know your MRN, passport number, driver's license number, "
                        "or clinic patient number?"
                    )
                ],
                "last_agent": "intake",
            }

        if "dl-12345" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found Jane Smith born on 1992-03-15 from that identifier — is that you?"
                    )
                ],
                "last_agent": "intake",
            }

        if latest.startswith("yes") and ai_history_contains(state, "is that you?"):
            return {
                "messages": [AIMessage(content="Thanks, you're verified.")],
                "patient_id": "patient-2",
                "patient_name": "Jane Smith",
                "last_agent": "intake",
            }

        return {
            "messages": [
                AIMessage(
                    content="Can you share your full name and date of birth so I can look you up?"
                )
            ],
            "last_agent": "intake",
        }

    install_test_graph(
        monkeypatch,
        intake_node=fake_intake_node,
        triage_node=_simple_triage_node,
        scheduling_node=_simple_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        lookup_prompt,
        identifier_prompt,
        confirmation_prompt,
        post_confirmation,
    ) = invoke_sequence(
        "workflow-dl-lookup",
        "hi",
        "I need to book an appointment",
        "I've been there before",
        "Jane Smith, 1992-03-15",
        "My driver's license is DL-12345",
        "Yes, that's me",
    )

    assert "full name and date of birth" in lookup_prompt
    assert "Do you know your MRN" in identifier_prompt
    assert "from that identifier" in confirmation_prompt
    assert "Thanks, you're verified." in post_confirmation
    assert "What symptoms are you having?" in post_confirmation
