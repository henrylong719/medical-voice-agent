from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from pytest import MonkeyPatch

from app.agent import supervisor
from app.agent.state import AgentState


def _state(**overrides: Any) -> AgentState:
    base: dict[str, Any] = {
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="Hello! Welcome to the university health clinic."),
            HumanMessage(content="I need to move my appointment"),
        ],
        "patient_id": "patient-1",
        "patient_name": "Sarah",
        "symptoms": [],
        "specialty_id": None,
        "appointment_id": None,
        "selected_appointment_id": None,
        "current_agent": "supervisor",
        "intent": None,
        "last_agent": None,
    }
    base.update(overrides)
    return cast(AgentState, base)


def test_supervisor_greets_on_first_message() -> None:
    result = asyncio.run(
        supervisor.supervisor_node(
            cast(
                AgentState,
                {
                    "messages": [HumanMessage(content="hi")],
                    "patient_id": None,
                    "patient_name": None,
                    "symptoms": [],
                    "specialty_id": None,
                    "appointment_id": None,
                    "selected_appointment_id": None,
                    "current_agent": "supervisor",
                    "intent": None,
                    "last_agent": None,
                },
            )
        )
    )

    assert result["current_agent"] == "done"
    assert "Welcome to the university health clinic" in result["messages"][0].content


def test_classify_intent_returns_normalized_label(monkeypatch: MonkeyPatch) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            return SimpleNamespace(content=[{"text": " ReSchedule "}])

    monkeypatch.setattr(supervisor, "ChatAnthropic", FakeLLM)

    result = asyncio.run(supervisor._classify_intent(_state()))

    assert result == "reschedule"


def test_classify_intent_returns_none_for_unknown_response(monkeypatch: MonkeyPatch) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            return SimpleNamespace(content="unknown")

    monkeypatch.setattr(supervisor, "ChatAnthropic", FakeLLM)

    result = asyncio.run(supervisor._classify_intent(_state()))

    assert result is None


def test_supervisor_routes_to_intake_when_patient_is_missing() -> None:
    result = asyncio.run(
        supervisor.supervisor_node(_state(patient_id=None, intent="reschedule"))
    )

    assert result == {"current_agent": "intake"}


def test_supervisor_routes_uin_correction_back_to_intake(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_classify(state: AgentState) -> str | None:
        return None

    monkeypatch.setattr(supervisor, "_classify_intent", fake_classify)

    result = asyncio.run(
        supervisor.supervisor_node(
            _state(
                intent="book",
                appointment_id="appt-1",
                selected_appointment_id="appt-1",
                messages=[
                    HumanMessage(content="hi"),
                    AIMessage(
                        content="Hello! Welcome to the university health clinic."
                    ),
                    HumanMessage(content="I have headaches"),
                    AIMessage(content="I found Henry Long on file — is that you?"),
                    HumanMessage(content="no sorry my uin is 123456787"),
                ],
            )
        )
    )

    assert result == {
        "patient_id": None,
        "patient_name": None,
        "appointment_id": None,
        "selected_appointment_id": None,
        "current_agent": "intake",
        "last_agent": None,
    }


def test_supervisor_re_runs_after_classifying_intent(monkeypatch: MonkeyPatch) -> None:
    async def fake_classify(state: AgentState) -> str | None:
        return "book"

    monkeypatch.setattr(supervisor, "_classify_intent", fake_classify)

    result = asyncio.run(supervisor.supervisor_node(_state()))

    assert result == {
        "intent": "book",
        "selected_appointment_id": None,
        "current_agent": "supervisor",
        "last_agent": None,
    }


def test_supervisor_asks_for_intent_when_still_unclear(monkeypatch: MonkeyPatch) -> None:
    async def fake_classify(state: AgentState) -> str | None:
        return None

    monkeypatch.setattr(supervisor, "_classify_intent", fake_classify)

    result = asyncio.run(supervisor.supervisor_node(_state()))

    assert result["current_agent"] == "done"
    assert "Are you looking to book a new appointment" in result["messages"][0].content
    assert "book a new appointment" in result["messages"][0].content


def test_supervisor_routes_booking_flow_based_on_specialty() -> None:
    triage = asyncio.run(
        supervisor.supervisor_node(
            _state(
                intent="book",
                messages=[
                    HumanMessage(content="hi"),
                    AIMessage(content="Hello! Welcome to the university health clinic."),
                    HumanMessage(content="I'd like to make an appointment"),
                ],
            )
        )
    )
    scheduling = asyncio.run(
        supervisor.supervisor_node(
            _state(
                intent="book",
                specialty_id="spec-cardio",
                messages=[
                    HumanMessage(content="hi"),
                    AIMessage(content="Hello! Welcome to the university health clinic."),
                    HumanMessage(content="I'd like to make an appointment"),
                ],
            )
        )
    )

    assert triage == {"current_agent": "triage"}
    assert scheduling == {"current_agent": "scheduling"}


def test_supervisor_routes_reschedule_and_cancel_to_scheduling() -> None:
    reschedule = asyncio.run(
        supervisor.supervisor_node(
            _state(
                intent="reschedule",
                messages=[
                    HumanMessage(content="hi"),
                    AIMessage(content="Hello! Welcome to the university health clinic."),
                    HumanMessage(content="I need to move my appointment"),
                ],
            )
        )
    )
    cancel = asyncio.run(
        supervisor.supervisor_node(
            _state(
                intent="cancel",
                messages=[
                    HumanMessage(content="hi"),
                    AIMessage(content="Hello! Welcome to the university health clinic."),
                    HumanMessage(content="I need to cancel my appointment"),
                ],
            )
        )
    )

    assert reschedule == {"current_agent": "scheduling"}
    assert cancel == {"current_agent": "scheduling"}


def test_supervisor_detects_explicit_intent_change_mid_flow() -> None:
    state = _state(
        intent="book",
        last_agent="triage",
        messages=[
            HumanMessage(content="hi"),
            AIMessage(content="Hello! Welcome to the university health clinic."),
            HumanMessage(content="I'd like to make an appointment"),
            AIMessage(content="Do you have a particular type of specialist in mind?"),
            HumanMessage(content="sorry actually i'd like to reschedule an appointment"),
        ],
        symptoms=["headache"],
        specialty_id="spec-neuro",
    )

    result = asyncio.run(supervisor.supervisor_node(state))

    assert result == {
        "intent": "reschedule",
        "symptoms": [],
        "specialty_id": None,
        "selected_appointment_id": None,
        "current_agent": "supervisor",
        "last_agent": None,
    }
