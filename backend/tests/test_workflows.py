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


async def _booking_triage_node(state: AgentState) -> dict:
    latest = latest_human_text(state).lower()

    if "headache" in latest or "dizziness" in latest:
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


async def _booking_scheduling_node(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "Do you have a preferred day or week in mind, or would you like the earliest available?"
                )
            )
        ],
        "last_agent": "scheduling",
    }


def test_booking_workflow_can_be_exercised_end_to_end_without_postman(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()
        patient_status = state.get("patient_status")

        if patient_status == "new":
            if "555-0100" in latest or "10/26/1985" in latest:
                return {
                    "messages": [
                        AIMessage(content="Thanks, you're registered as Sarah Connor.")
                    ],
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

    install_test_graph(
        monkeypatch,
        intake_node=fake_intake_node,
        triage_node=_booking_triage_node,
        scheduling_node=_booking_scheduling_node,
    )

    greeting, status_question, registration_prompt, post_registration, symptoms_prompt = invoke_sequence(
        "workflow-booking",
        "hi",
        "I need to book an appointment",
        "This is my first visit",
        "Sarah Connor, 10/26/1985, 555-0100",
        "I've had headaches and dizziness",
    )

    assert greeting == "Hello! Welcome to the clinic. How can I help you today?"
    assert (
        status_question
        == "Have you been seen at this clinic before, or is this your first visit?"
    )
    assert (
        registration_prompt
        == "What is your full name, date of birth, and phone number?"
    )
    assert "Thanks, you're registered as Sarah Connor." in post_registration
    assert "What symptoms are you having?" in post_registration
    assert "Neurology seems like the right specialty." in symptoms_prompt
    assert "preferred day or week" in symptoms_prompt


def test_returning_patient_unique_match_requires_confirmation_before_triage(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "sarah connor" in latest and "1985-10-26" in latest:
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

    install_test_graph(
        monkeypatch,
        intake_node=fake_intake_node,
        triage_node=_booking_triage_node,
        scheduling_node=_booking_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        lookup_prompt,
        confirmation_prompt,
        post_confirmation,
        booking_prompt,
    ) = invoke_sequence(
        "workflow-returning-unique",
        "hi",
        "I need to book an appointment",
        "I've been seen there before",
        "Sarah Connor, 1985-10-26",
        "Yes, that's me",
        "I've had recurring headaches",
    )

    assert "full name and date of birth" in lookup_prompt
    assert "Sarah Connor born on 1985-10-26" in confirmation_prompt
    assert "Thanks, you're verified." in post_confirmation
    assert "What symptoms are you having?" in post_confirmation
    assert "Neurology seems like the right specialty." in booking_prompt
    assert "preferred day or week" in booking_prompt


def test_returning_patient_can_resolve_ambiguous_demographics_with_phone(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "alex kim" in latest and "1990-01-02" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found more than one record with that name and date of birth. What phone number do you have on file?"
                    )
                ],
                "last_agent": "intake",
            }

        if "555-0101" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found Alex Kim born on 1990-01-02 ending in phone 0101 on file — is that you?"
                    )
                ],
                "last_agent": "intake",
            }

        if latest.startswith("yes") and ai_history_contains(state, "is that you?"):
            return {
                "messages": [AIMessage(content="Thanks, you're verified.")],
                "patient_id": "patient-2",
                "patient_name": "Alex Kim",
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
        triage_node=_booking_triage_node,
        scheduling_node=_booking_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        lookup_prompt,
        phone_prompt,
        confirmation_prompt,
        post_confirmation,
    ) = invoke_sequence(
        "workflow-phone-resolution",
        "hi",
        "I need to book an appointment",
        "Yes, I've been there before",
        "Alex Kim, 1990-01-02",
        "555-0101",
        "Yes, that's me",
    )

    assert "full name and date of birth" in lookup_prompt
    assert "What phone number do you have on file?" in phone_prompt
    assert "ending in phone 0101" in confirmation_prompt
    assert "Thanks, you're verified." in post_confirmation
    assert "What symptoms are you having?" in post_confirmation


def test_returning_patient_can_fallback_to_identifier_after_no_match(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "sarah connor" in latest and "1985-10-26" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I couldn't find a record with that name and date of birth. Do you know your MRN, passport number, driver's license number, or clinic patient number?"
                    )
                ],
                "last_agent": "intake",
            }

        if "mrn-1001" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found Sarah Connor born on 1985-10-26 from that identifier — is that you?"
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

    install_test_graph(
        monkeypatch,
        intake_node=fake_intake_node,
        triage_node=_booking_triage_node,
        scheduling_node=_booking_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        lookup_prompt,
        identifier_prompt,
        confirmation_prompt,
        post_confirmation,
    ) = invoke_sequence(
        "workflow-identifier-fallback",
        "hi",
        "I need to book an appointment",
        "I've been there before",
        "Sarah Connor, 1985-10-26",
        "My MRN is MRN-1001",
        "Yes, that's me",
    )

    assert "full name and date of birth" in lookup_prompt
    assert "Do you know your MRN" in identifier_prompt
    assert "from that identifier" in confirmation_prompt
    assert "Thanks, you're verified." in post_confirmation
    assert "What symptoms are you having?" in post_confirmation


def test_returning_patient_without_identifier_is_offered_registration(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "sarah connor" in latest and "1985-10-26" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I couldn't find a record with that name and date of birth. Do you know your MRN, passport number, driver's license number, or clinic patient number?"
                    )
                ],
                "last_agent": "intake",
            }

        if "don't know" in latest or "dont know" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I couldn't find an existing record. If you'd like, I can register you as a new patient."
                    )
                ],
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
        triage_node=_booking_triage_node,
        scheduling_node=_booking_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        lookup_prompt,
        identifier_prompt,
        registration_offer,
    ) = invoke_sequence(
        "workflow-offer-registration",
        "hi",
        "I need to book an appointment",
        "I've been there before",
        "Sarah Connor, 1985-10-26",
        "I don't know any identifier",
    )

    assert "full name and date of birth" in lookup_prompt
    assert "Do you know your MRN" in identifier_prompt
    assert "register you as a new patient" in registration_offer


def test_returning_patient_ambiguous_match_escalates_cleanly(
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_intake_node(state: AgentState) -> dict:
        latest = latest_human_text(state).lower()

        if "alex kim" in latest and "1990-01-02" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I found more than one record with that name and date of birth. What phone number do you have on file?"
                    )
                ],
                "last_agent": "intake",
            }

        if "555-0000" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I still found multiple records. Do you know your MRN, passport number, driver's license number, or another clinic patient number?"
                    )
                ],
                "last_agent": "intake",
            }

        if "i don't know" in latest or "dont know" in latest:
            return {
                "messages": [
                    AIMessage(
                        content="I can't safely verify which record is yours, so a staff member will need to help."
                    )
                ],
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

    async def fake_triage_node(state: AgentState) -> dict:
        return {"last_agent": "triage"}

    async def fake_scheduling_node(state: AgentState) -> dict:
        return {"last_agent": "scheduling"}

    install_test_graph(
        monkeypatch,
        intake_node=fake_intake_node,
        triage_node=fake_triage_node,
        scheduling_node=fake_scheduling_node,
    )

    (
        _greeting,
        _status_question,
        prompt,
        phone_request,
        identifier_request,
        handoff,
    ) = invoke_sequence(
        "workflow-ambiguous",
        "hello",
        "I need to book an appointment",
        "I've been there before",
        "Alex Kim, 1990-01-02",
        "555-0000",
        "I don't know any identifier",
    )

    assert "full name and date of birth" in prompt
    assert "What phone number do you have on file?" in phone_request
    assert "Do you know your MRN" in identifier_request
    assert "staff member will need to help" in handoff
