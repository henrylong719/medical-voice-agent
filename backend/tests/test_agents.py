from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pytest import MonkeyPatch

from app.agent import agents
from app.agent.state import AgentState
from tests.support import FakeAsyncAgent


def _base_state() -> AgentState:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="Hello")],
        "patient_id": None,
        "patient_name": None,
        "symptoms": [],
        "specialty_id": None,
        "appointment_id": None,
        "selected_appointment_id": None,
        "current_agent": "supervisor",
        "intent": None,
        "last_agent": None,
    }
    return cast(AgentState, state)


def test_intake_node_extracts_registered_patient_identity(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(content="I can get that set up for you."),
        ToolMessage(
            content=(
                "Successfully registered Sarah Connor "
                "(ID: patient-1, UIN: 123456789)."
            ),
            tool_call_id="call-1",
        ),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert fake_agent.calls == [{"messages": state["messages"]}]
    assert result["messages"] == new_messages[1:]
    assert result["patient_id"] == "patient-1"
    assert result["patient_name"] == "Sarah Connor"


def test_intake_node_waits_for_patient_confirmation_before_setting_identity(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(content="Thanks! Let me look that up."),
        ToolMessage(
            content=(
                "Patient found: Sarah Connor "
                "(ID: patient-1, UIN: 123456789). "
                "Phone: not on file. Allergies: None listed."
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found Sarah Connor on file — is that you?"),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"] == new_messages[1:]
    assert result["patient_id"] is None
    assert result["patient_name"] is None


def test_intake_node_promotes_lookup_after_patient_confirms(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["messages"] = [
        HumanMessage(content="I have headaches"),
        AIMessage(
            content=(
                "To get you set up, could I have your 9-digit university "
                "ID number?"
            )
        ),
        HumanMessage(content="123456789"),
        AIMessage(content="Thanks! Let me look that up."),
        ToolMessage(
            content=(
                "Patient found: Sarah Connor "
                "(ID: patient-1, UIN: 123456789). "
                "Phone: not on file. Allergies: None listed."
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found Sarah Connor on file — is that you?"),
        HumanMessage(content="yes that's me"),
    ]
    returned_messages = state["messages"] + [
        AIMessage(content="Thank you. Let's keep going."),
    ]
    fake_agent = FakeAsyncAgent({"messages": returned_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"] == returned_messages[len(state["messages"]) :]
    assert result["patient_id"] == "patient-1"
    assert result["patient_name"] == "Sarah Connor"


def test_intake_prompt_requires_phone_before_registration() -> None:
    assert "full name and phone number" in agents._INTAKE_PROMPT
    assert (
        "Do NOT call register_patient until you have BOTH their full name and "
        "phone number"
        in agents._INTAKE_PROMPT
    )
    assert (
        "Always collect both the patient's full name and phone number "
        "during new registration."
        in agents._INTAKE_PROMPT
    )
    assert (
        "Accept any non-empty phone number string exactly as the patient "
        "provides it."
        in agents._INTAKE_PROMPT
    )
    assert (
        "Only call register_patient AFTER the patient confirms the phone "
        "number is right."
        in agents._INTAKE_PROMPT
    )
    assert "Read the phone number back slowly and confirm it before saving." in agents._INTAKE_PROMPT


def test_triage_node_extracts_specialty_and_symptoms(monkeypatch: MonkeyPatch) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(
            content="Let me check that.",
            tool_calls=[
                {
                    "name": "triage_symptoms",
                    "args": {"symptoms": ["chest pain", "dizziness"]},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=(
                "Hybrid triage results\n"
                "- Cardiology (ID: spec-cardio): score 8.50, matched on: chest pain"
            ),
            tool_call_id="call-1",
        ),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_triage_agent", lambda: fake_agent)

    result = asyncio.run(agents.triage_node(state))

    assert result["messages"] == new_messages[1:]
    assert result["specialty_id"] == "spec-cardio"
    assert result["symptoms"] == ["chest pain", "dizziness"]


def test_scheduling_node_extracts_booked_appointment_id(monkeypatch: MonkeyPatch) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(content="Booking that now."),
        ToolMessage(
            content=(
                "Appointment booked successfully! "
                "Appointment ID: appt-1. "
                "Scheduled for Monday, April 13th at 9:00 AM."
            ),
            tool_call_id="call-1",
        ),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_scheduling_agent", lambda: fake_agent)

    result = asyncio.run(agents.scheduling_node(state))

    assert result["messages"] == new_messages[1:]
    assert result["appointment_id"] == "appt-1"
    assert result["selected_appointment_id"] is None
    assert result["intent"] is None


def test_scheduling_node_ignores_stale_booking_tool_messages(monkeypatch: MonkeyPatch) -> None:
    state = _base_state()
    state["intent"] = "reschedule"
    state["messages"] = [
        HumanMessage(content="Book me an appointment"),
        ToolMessage(
            content=(
                "Appointment booked successfully! "
                "Appointment ID: appt-old. "
                "Scheduled for Monday, April 13th at 9:00 AM."
            ),
            tool_call_id="call-old",
        ),
        HumanMessage(content="I need to move it"),
    ]

    returned_messages = state["messages"] + [
        AIMessage(
            content=(
                "I see you have an appointment with Dr. Rodriguez on Tuesday, "
                "April 14th at 11:00 AM. Is that the one you'd like to reschedule?"
            )
        )
    ]
    fake_agent = FakeAsyncAgent({"messages": returned_messages})
    monkeypatch.setattr(agents, "_get_scheduling_agent", lambda: fake_agent)

    result = asyncio.run(agents.scheduling_node(state))

    assert result["messages"] == returned_messages[len(state["messages"]) :]
    assert result["appointment_id"] is None
    assert result["selected_appointment_id"] is None
    assert "intent" not in result


def test_scheduling_node_tracks_selected_appointment_for_reschedule_preview(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["intent"] = "reschedule"
    new_messages = state["messages"] + [
        AIMessage(
            content="Let me look for another time with the same doctor.",
            tool_calls=[
                {
                    "name": "reschedule_appointment",
                    "args": {
                        "appointment_id": "appt-1",
                        "patient_id": "patient-1",
                        "preferred_day": "next week",
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=(
                "Current appointment: Dr. Rodriguez (Neurology) on Monday, "
                "April 13th at 8:00 AM (appointment_id: appt-1, "
                "specialty_id: spec-neuro).\n"
                "The current appointment has NOT been cancelled."
            ),
            tool_call_id="call-1",
        ),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_scheduling_agent", lambda: fake_agent)

    result = asyncio.run(agents.scheduling_node(state))

    assert result["messages"] == new_messages[1:]
    assert result["appointment_id"] is None
    assert result["selected_appointment_id"] == "appt-1"
    assert "intent" not in result


def test_scheduling_node_marks_completed_reschedule(monkeypatch: MonkeyPatch) -> None:
    state = _base_state()
    state["intent"] = "reschedule"
    new_messages = state["messages"] + [
        AIMessage(
            content="Rescheduling that now.",
            tool_calls=[
                {
                    "name": "reschedule_appointment",
                    "args": {
                        "appointment_id": "appt-1",
                        "patient_id": "patient-1",
                        "new_doctor_id": "doctor-1",
                        "new_specialty_id": "spec-neuro",
                        "new_start_at": "2026-04-20T14:00:00+00:00",
                        "new_end_at": "2026-04-20T14:45:00+00:00",
                    },
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=(
                "Appointment rescheduled successfully! "
                "Old appointment on Monday, April 13th at 9:00 AM was moved to "
                "Monday, April 20th at 9:00 AM. Appointment ID: appt-1."
            ),
            tool_call_id="call-2",
        ),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_scheduling_agent", lambda: fake_agent)

    result = asyncio.run(agents.scheduling_node(state))

    assert result["messages"] == new_messages[1:]
    assert result["appointment_id"] == "appt-1"
    assert result["selected_appointment_id"] == "appt-1"
    assert result["intent"] is None


def test_scheduling_prompt_says_unavailable_bucket_before_alternatives() -> None:
    assert "If the requested morning/afternoon bucket has no matches, say that before offering alternatives." in agents._SCHEDULING_PROMPT
    assert "Do NOT list times from a different bucket until the patient asks for them or agrees to switch." in agents._SCHEDULING_PROMPT
    assert "Always include the current patient's ID in find_appointment, reschedule_appointment, and cancel_appointment." in agents._SCHEDULING_PROMPT
