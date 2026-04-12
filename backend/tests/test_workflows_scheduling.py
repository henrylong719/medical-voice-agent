from __future__ import annotations

from langchain_core.messages import AIMessage
from pytest import MonkeyPatch

from app.agent.state import AgentState
from tests.workflow_support import (
    ai_history_contains,
    install_test_graph,
    invoke_sequence,
    latest_human_text,
)


def _make_verified_returning_intake():
    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "sarah connor" in latest and (
            "1985-10-26" in latest or "10/26/1985" in latest
        ):
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

        return {
            "messages": [
                AIMessage(
                    content="Can you share your full name and date of birth so I can look you up?"
                )
            ],
            "last_agent": "intake",
        }

    return fake_intake_node


async def _unused_triage_node(state: AgentState) -> dict:
    return {"last_agent": "triage"}


def test_reschedule_workflow_skips_visit_status_and_reaches_preview(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_scheduling_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if latest.startswith("yes") and ai_history_contains(
            state, "is that the one you'd like to reschedule?"
        ):
            return {
                "messages": [
                    AIMessage(
                        content="I have openings on Monday, April 27th and Tuesday, April 28th. Which day works best?"
                    )
                ],
                "selected_appointment_id": "appt-1",
                "last_agent": "scheduling",
            }

        if "neurology" in latest and ai_history_contains(
            state, "which doctor or specialty"
        ):
            return {
                "messages": [
                    AIMessage(
                        content="I see your neurology appointment with Dr. Rodriguez on Thursday, April 23rd. Is that the one you'd like to reschedule?"
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
        intake_node=_make_verified_returning_intake(),
        triage_node=_unused_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        _greeting,
        intake_prompt,
        confirmation_prompt,
        scheduling_prompt,
        appointment_prompt,
        preview_prompt,
    ) = invoke_sequence(
        "workflow-reschedule",
        "hi",
        "I need to reschedule my appointment",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
        "Neurology",
        "Yes, that's the one",
    )

    assert "Have you been seen at this clinic before" not in intake_prompt
    assert "full name and date of birth" in intake_prompt
    assert "Sarah Connor born on 1985-10-26" in confirmation_prompt
    assert "Thanks, you're verified." in scheduling_prompt
    assert "which doctor or specialty" in scheduling_prompt
    assert "Is that the one you'd like to reschedule?" in appointment_prompt
    assert "Which day works best?" in preview_prompt


def test_cancel_workflow_skips_visit_status_and_requires_confirmation(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_scheduling_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "dr. rodriguez" in latest and ai_history_contains(
            state, "which doctor or specialty"
        ):
            return {
                "messages": [
                    AIMessage(
                        content="I see your appointment with Dr. Rodriguez on Thursday, April 23rd. Would you like to cancel this one?"
                    )
                ],
                "appointment_id": "appt-1",
                "last_agent": "scheduling",
            }

        if "cancel" in latest and ai_history_contains(
            state, "would you like to cancel this one?"
        ):
            return {
                "messages": [
                    AIMessage(
                        content="Your appointment has been cancelled. Can I help with anything else?"
                    )
                ],
                "appointment_id": "appt-1",
                "intent": None,
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
        intake_node=_make_verified_returning_intake(),
        triage_node=_unused_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        _greeting,
        intake_prompt,
        confirmation_prompt,
        scheduling_prompt,
        cancel_confirmation,
        cancellation_result,
    ) = invoke_sequence(
        "workflow-cancel",
        "hi",
        "I need to cancel my appointment",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
        "Dr. Rodriguez",
        "Yes, cancel it",
    )

    assert "Have you been seen at this clinic before" not in intake_prompt
    assert "full name and date of birth" in intake_prompt
    assert "Sarah Connor born on 1985-10-26" in confirmation_prompt
    assert "which doctor or specialty" in scheduling_prompt
    assert "Would you like to cancel this one?" in cancel_confirmation
    assert "appointment has been cancelled" in cancellation_result


def test_booking_flow_can_switch_mid_conversation_to_reschedule(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_triage_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()
        if "headache" in latest:
            return {
                "messages": [
                    AIMessage(content="Neurology seems like the right specialty.")
                ],
                "specialty_id": "spec-neuro",
                "last_agent": "triage",
            }
        return {
            "messages": [AIMessage(content="What symptoms are you having?")],
            "last_agent": "triage",
        }

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
        intake_node=_make_verified_returning_intake(),
        triage_node=fake_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        _lookup_prompt,
        _confirmation_prompt,
        triage_prompt,
        switched_prompt,
    ) = invoke_sequence(
        "workflow-intent-switch",
        "hi",
        "I need to book an appointment",
        "I've been seen there before",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
        "Actually, I need to reschedule instead.",
    )

    assert "What symptoms are you having?" in triage_prompt
    assert "which doctor or specialty" in switched_prompt
    assert "What symptoms are you having?" not in switched_prompt
