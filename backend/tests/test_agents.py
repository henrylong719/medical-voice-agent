from __future__ import annotations

import asyncio
import json
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
        "patient_status": None,
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
            content=json.dumps(
                {
                    "status": "registered",
                    "message": "Patient registered successfully.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone_last4": "0100",
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert fake_agent.calls == [{"messages": state["messages"]}]
    assert getattr(result["messages"][0], "type", None) == "tool"
    assert result["messages"][-1].content == "Perfect! You're all registered."
    assert result["patient_id"] == "patient-1"
    assert result["patient_name"] == "Sarah Connor"


def test_intake_node_replaces_registration_handoff_with_concise_confirmation(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(
            content="Let me get you registered.",
            tool_calls=[
                {
                    "name": "register_patient",
                    "args": {
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone": "555-0100",
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "registered",
                    "message": "Patient registered successfully.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone_last4": "0100",
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="Perfect! You're all registered. A staff member will be with you shortly."),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert getattr(result["messages"][0], "type", None) == "ai"
    assert result["messages"][0].tool_calls[0]["name"] == "register_patient"
    assert result["messages"][-1].content == "Perfect! You're all registered."
    assert all(
        "staff member" not in getattr(msg, "content", "")
        for msg in result["messages"]
        if getattr(msg, "type", None) == "ai"
    )


def test_intake_node_reuses_existing_record_after_registration_attempt(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(
            content="Let me check that registration.",
            tool_calls=[
                {
                    "name": "register_patient",
                    "args": {
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone": "555-0100",
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "already_exists",
                    "message": "A matching patient already exists.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone_last4": "0100",
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="You're already in our system. A staff member will be with you shortly."),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["patient_id"] == "patient-1"
    assert result["patient_name"] == "Sarah Connor"
    assert result["messages"][-1].content == "You're already in our system. You're all set."
    assert all(
        "staff member" not in getattr(msg, "content", "")
        for msg in result["messages"]
        if getattr(msg, "type", None) == "ai"
    )


def test_intake_node_waits_for_patient_confirmation_before_setting_identity(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(content="Thanks! Let me look that up."),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "single_match",
                    "message": "One patient matched those demographics.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone_last4": None,
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found Sarah Connor born on 1985-10-26 on file — is that you?"),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"][0].content == json.dumps(
        {
            "status": "single_match",
            "message": "One patient matched those demographics.",
            "patient": {
                "patient_id": "patient-1",
                "full_name": "Sarah Connor",
                "date_of_birth": "1985-10-26",
                "phone_last4": None,
                "email": None,
                "allergies": [],
            },
        }
    )
    assert result["messages"][-1].content == "I found Sarah Connor, born October 26, 1985. Is that you?"
    assert result["patient_id"] is None
    assert result["patient_name"] is None


def test_intake_node_promotes_lookup_after_patient_confirms(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["messages"] = [
        HumanMessage(content="I have headaches"),
        AIMessage(
            content="Can you share your full name and date of birth so I can look you up?"
        ),
        HumanMessage(content="Sarah Connor, October 26 1985"),
        AIMessage(content="Thanks! Let me look that up."),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "single_match",
                    "message": "One patient matched those demographics.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone_last4": None,
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found Sarah Connor born on 1985-10-26 on file — is that you?"),
        HumanMessage(content="yes that's me"),
    ]
    returned_messages = state["messages"] + [
        AIMessage(content="Thank you. Let's keep going."),
    ]
    fake_agent = FakeAsyncAgent({"messages": returned_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"][-1].content == "Great! You're verified."
    assert result["patient_id"] == "patient-1"
    assert result["patient_name"] == "Sarah Connor"


def test_intake_node_promotes_lookup_after_llm_confirmed_identity_reply(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["messages"] = [
        HumanMessage(content="I have headaches"),
        AIMessage(
            content="Can you share your full name and date of birth so I can look you up?"
        ),
        HumanMessage(content="Sarah Connor, October 26 1985"),
        AIMessage(content="Thanks! Let me look that up."),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "single_match",
                    "message": "One patient matched those demographics.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Sarah Connor",
                        "date_of_birth": "1985-10-26",
                        "phone_last4": None,
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found Sarah Connor born on 1985-10-26 on file — is that you?"),
        HumanMessage(content="It is me"),
    ]
    returned_messages = state["messages"] + [
        AIMessage(content="Thank you. Let's keep going."),
    ]
    fake_agent = FakeAsyncAgent({"messages": returned_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    async def fake_classify_identity_reply_with_llm(
        content: object,
    ) -> str | None:
        assert content == "It is me"
        return "affirmative"

    monkeypatch.setattr(
        agents,
        "_classify_identity_reply_with_llm",
        fake_classify_identity_reply_with_llm,
    )

    result = asyncio.run(agents.intake_node(state))

    assert result["patient_id"] == "patient-1"
    assert result["patient_name"] == "Sarah Connor"


def test_intake_node_replaces_verification_handoff_with_concise_confirmation(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["messages"] = [
        HumanMessage(content="Alice Johnson, May 14 1992"),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "single_match",
                    "message": "One patient matched those demographics.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Alice Johnson",
                        "date_of_birth": "1992-05-14",
                        "phone_last4": "1001",
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found Alice Johnson born on 1992-05-14 on file — is that you?"),
        HumanMessage(content="Yes, that's me"),
    ]
    returned_messages = state["messages"] + [
        AIMessage(content="Perfect, you're verified. A staff member will be with you shortly to help."),
    ]
    fake_agent = FakeAsyncAgent({"messages": returned_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"][-1].content == "Great! You're verified."
    assert result["patient_id"] == "patient-1"


def test_intake_node_suppresses_redundant_verification_message_when_flow_continues(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["intent"] = "book"
    state["messages"] = [
        HumanMessage(content="Alice Johnson, May 14 1992"),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "single_match",
                    "message": "One patient matched those demographics.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Alice Johnson",
                        "date_of_birth": "1992-05-14",
                        "phone_last4": "1001",
                        "email": None,
                        "allergies": [],
                    },
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found Alice Johnson born on 1992-05-14 on file — is that you?"),
        HumanMessage(content="Yes, that's me"),
    ]
    returned_messages = state["messages"] + [
        AIMessage(content="You're verified. Thank you!"),
    ]
    fake_agent = FakeAsyncAgent({"messages": returned_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"] == []
    assert result["patient_id"] == "patient-1"
    assert result["patient_name"] == "Alice Johnson"


def test_classify_identity_reply_with_llm_returns_affirmative(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]):
            return type("Response", (), {"content": [{"text": " affirmative "}]} )()

    monkeypatch.setattr(agents, "ChatAnthropic", FakeLLM)

    result = asyncio.run(agents._classify_identity_reply_with_llm("It is me"))

    assert result == "affirmative"


def test_classify_identity_reply_with_llm_returns_none_for_unknown(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]):
            return type("Response", (), {"content": "unknown"})()

    monkeypatch.setattr(agents, "ChatAnthropic", FakeLLM)

    result = asyncio.run(agents._classify_identity_reply_with_llm("maybe"))

    assert result is None


def test_intake_node_keeps_identity_unset_for_multiple_demographic_matches(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(content="Let me check our records."),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "multiple_matches",
                    "message": (
                        "Multiple patients matched those demographics. Ask for a phone "
                        "number first, then try demographics again before asking for "
                        "a stronger identifier."
                    ),
                    "candidates": [
                        {
                            "patient_id": "patient-1",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone_last4": "0100",
                            "email": None,
                            "allergies": [],
                        },
                        {
                            "patient_id": "patient-2",
                            "full_name": "Alex Kim",
                            "date_of_birth": "1990-01-02",
                            "phone_last4": "0101",
                            "email": None,
                            "allergies": [],
                        },
                    ],
                    "match_count": 2,
                    "lookup_method": "demographics",
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I found more than one record with that name and date of birth. What phone number do you have on file?"),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"] == new_messages[1:]
    assert result["patient_id"] is None
    assert result["patient_name"] is None


def test_intake_node_keeps_identity_unset_for_no_match_returning_lookup(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    new_messages = state["messages"] + [
        AIMessage(content="Let me check our records."),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "no_match",
                    "message": (
                        "No patient matched that name and date of birth. Ask for a stronger "
                        "identifier such as MRN, passport number, driver's license number, "
                        "or clinic patient number. If they do not know one, then offer "
                        "registration if this may be their first visit."
                    ),
                    "lookup_method": "demographics",
                }
            ),
            tool_call_id="call-1",
        ),
        AIMessage(content="I couldn’t find a record with that name and date of birth. Do you know your MRN, passport number, driver's license number, or clinic patient number?"),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["messages"] == new_messages[1:]
    assert result["patient_id"] is None
    assert result["patient_name"] is None


def test_intake_node_clarifies_identifier_match_after_demographics_discrepancy(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["messages"] = [
        HumanMessage(content="Emma Davis"),
        AIMessage(content="What is your date of birth?"),
        HumanMessage(content="January 31, 1990"),
        AIMessage(content="Let me check that."),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "no_match",
                    "message": "No patient matched that name and date of birth.",
                    "lookup_method": "demographics",
                }
            ),
            tool_call_id="call-demographics",
        ),
        AIMessage(content="Do you know your clinic patient number?"),
        HumanMessage(content="My clinic patient number is CLINIC-9005"),
    ]
    new_messages = state["messages"] + [
        AIMessage(
            content="Let me look that up.",
            tool_calls=[
                {
                    "name": "find_patient_by_identifier",
                    "args": {
                        "identifier_type": "clinic_patient_number",
                        "identifier_value": "CLINIC-9005",
                    },
                    "id": "call-identifier",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "single_match",
                    "message": "One patient matched that identifier.",
                    "patient": {
                        "patient_id": "patient-1",
                        "full_name": "Emma Davis",
                        "date_of_birth": "1990-01-30",
                        "phone_last4": "1005",
                        "email": "emma.d@example.com",
                        "allergies": [],
                    },
                    "lookup_method": "identifier",
                    "matched_identifier": {
                        "identifier_type": "clinic_patient_number",
                        "identifier_value_masked": "*******9005",
                    },
                }
            ),
            tool_call_id="call-identifier",
        ),
        AIMessage(content="Perfect! I found your record. Is that you?"),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_intake_agent", lambda: fake_agent)

    result = asyncio.run(agents.intake_node(state))

    assert result["patient_id"] is None
    assert result["patient_name"] is None
    assert result["messages"][-1].content == (
        "I found a record for Emma Davis using that identifier. "
        "The date of birth on file is January 30, 1990, which is different "
        "from what you told me earlier. "
        "Is that your record?"
    )


def test_intake_prompt_requires_phone_before_registration() -> None:
    assert "full name, date of birth, and phone number" in agents._INTAKE_PROMPT
    assert "find_patient_by_identifier" in agents._INTAKE_PROMPT
    assert "find_patients_by_demographics" in agents._INTAKE_PROMPT
    assert (
        "If the patient wants to reschedule or cancel an appointment, treat them as a "
        "returning patient immediately."
        in agents._INTAKE_PROMPT
    )
    assert (
        "Do NOT tell the patient they are registered unless register_patient has "
        "succeeded in this turn."
        in agents._INTAKE_PROMPT
    )
    assert (
        "Do NOT ask whether the patient is new or returning when they want to cancel or reschedule"
        in agents._INTAKE_PROMPT
    )
    assert (
        "Do NOT say a staff member will help, transfer the patient, or hand off the conversation"
        in agents._INTAKE_PROMPT
    )
    assert (
        "NEVER claim the patient is registered or verified unless the corresponding tool lookup/registration succeeded."
        in agents._INTAKE_PROMPT
    )
    assert (
        "If the patient says they have been seen here before, first collect their full "
        "name and date of birth and call find_patients_by_demographics."
        in agents._INTAKE_PROMPT
    )
    assert (
        "ask for the phone number and call find_patients_by_demographics "
        "again with it."
        in agents._INTAKE_PROMPT
    )
    assert (
        "ask whether they know an MRN, passport number, driver's license number"
        in agents._INTAKE_PROMPT
    )
    assert (
        "If demographic lookup still returns multiple matches after using the phone number"
        in agents._INTAKE_PROMPT
    )
    assert (
        "If demographic lookup returns no match and the patient believes they have been seen"
        in agents._INTAKE_PROMPT
    )
    assert (
        "If a strong identifier returns a single clear match, ask for explicit confirmation"
        in agents._INTAKE_PROMPT
    )
    assert (
        "If a strong identifier also fails to produce a single clear match"
        in agents._INTAKE_PROMPT
    )
    assert "Do NOT pretend the details matched." in agents._INTAKE_PROMPT
    assert (
        "offer to register them as a new patient"
        in agents._INTAKE_PROMPT
    )
    assert (
        "Do NOT call register_patient until you have all three."
        in agents._INTAKE_PROMPT
    )
    assert (
        "Always collect full name, date of birth, and phone number "
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
    assert "Do NOT guess which patient record is correct." in agents._INTAKE_PROMPT
    assert (
        "Start returning-patient lookup with full name and date of birth before asking for stronger identifiers."
        in agents._INTAKE_PROMPT
    )


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


def test_triage_node_extracts_specialty_id_from_list_specialties_confirmation(
    monkeypatch: MonkeyPatch,
) -> None:
    state = _base_state()
    state["messages"] = [HumanMessage(content="Yes, Neurology sounds right.")]
    new_messages = state["messages"] + [
        AIMessage(
            content="Let me check whether we offer neurology here.",
            tool_calls=[
                {
                    "name": "list_specialties",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=(
                "Available specialties:\n"
                "- Cardiology (ID: spec-cardio): Heart care\n"
                "- Neurology (ID: spec-neuro): Brain and nerves\n"
            ),
            tool_call_id="call-1",
        ),
        AIMessage(
            content=(
                "Great news — we do have a Neurology department! "
                "A staff member will be in touch shortly to schedule your appointment."
            )
        ),
    ]
    fake_agent = FakeAsyncAgent({"messages": new_messages})
    monkeypatch.setattr(agents, "_get_triage_agent", lambda: fake_agent)

    result = asyncio.run(agents.triage_node(state))

    assert result["specialty_id"] == "spec-neuro"
    assert all(
        "staff member" not in getattr(msg, "content", "")
        for msg in result["messages"]
        if getattr(msg, "type", None) == "ai"
    )


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
    assert "If book_appointment says the slot is no longer available or no longer bookable" in agents._SCHEDULING_PROMPT
