"""
Tests for supervisor helper functions — intent keyword detection,
identity correction detection, patient status classification, and
other internal logic not covered by the main supervisor_node tests.
"""
from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from pytest import MonkeyPatch

from app.agent import supervisor
from app.agent.state import AgentState


# ── Helpers ───────────────────────────────────────────────────

def _msg(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def _state(**overrides: Any) -> AgentState:
    base: dict[str, Any] = {
        "messages": [HumanMessage(content="hi")],
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
    base.update(overrides)
    return cast(AgentState, base)


# ============================================================
# _intent_keyword_from_message
# ============================================================

def test_intent_keyword_reschedule() -> None:
    assert supervisor._intent_keyword_from_message(_msg("I'd like to reschedule")) == "reschedule"


def test_intent_keyword_move_appointment() -> None:
    assert supervisor._intent_keyword_from_message(_msg("move my appointment")) == "reschedule"


def test_intent_keyword_different_time() -> None:
    assert supervisor._intent_keyword_from_message(_msg("I want a different time")) == "reschedule"


def test_intent_keyword_change_appointment() -> None:
    assert supervisor._intent_keyword_from_message(_msg("change my appointment")) == "reschedule"


def test_intent_keyword_cancel() -> None:
    assert supervisor._intent_keyword_from_message(_msg("cancel my appointment")) == "cancel"


def test_intent_keyword_call_off() -> None:
    assert supervisor._intent_keyword_from_message(_msg("call off my appointment")) == "cancel"


def test_intent_keyword_book() -> None:
    assert supervisor._intent_keyword_from_message(_msg("book an appointment")) == "book"


def test_intent_keyword_make() -> None:
    assert supervisor._intent_keyword_from_message(_msg("make an appointment")) == "book"


def test_intent_keyword_schedule() -> None:
    assert supervisor._intent_keyword_from_message(_msg("schedule an appointment")) == "book"


def test_intent_keyword_none_for_vague() -> None:
    assert supervisor._intent_keyword_from_message(_msg("hello there")) is None


def test_intent_keyword_none_for_empty() -> None:
    assert supervisor._intent_keyword_from_message(_msg("")) is None


def test_intent_keyword_reschedule_takes_priority_over_cancel() -> None:
    # "reschedule" appears before "cancel" in the check order
    result = supervisor._intent_keyword_from_message(
        _msg("I want to reschedule, not cancel")
    )
    assert result == "reschedule"


# ============================================================
# _looks_like_identity_correction
# ============================================================

def test_identity_correction_wrong_patient() -> None:
    assert supervisor._looks_like_identity_correction(_msg("That's the wrong patient")) is True


def test_identity_correction_not_me() -> None:
    assert supervisor._looks_like_identity_correction(_msg("That's not me")) is True


def test_identity_correction_my_mrn() -> None:
    assert supervisor._looks_like_identity_correction(_msg("My MRN is 12345")) is True


def test_identity_correction_passport() -> None:
    assert supervisor._looks_like_identity_correction(
        _msg("My passport number is AB123456")
    ) is True


def test_identity_correction_drivers_license() -> None:
    assert supervisor._looks_like_identity_correction(
        _msg("My driver's license is DL-99999")
    ) is True


def test_identity_correction_wrong_record() -> None:
    assert supervisor._looks_like_identity_correction(_msg("wrong record")) is True


def test_identity_correction_not_my_record() -> None:
    assert supervisor._looks_like_identity_correction(_msg("That's not my record")) is True


def test_identity_correction_wrong_date_of_birth() -> None:
    assert supervisor._looks_like_identity_correction(_msg("That's the wrong date of birth")) is True


def test_identity_correction_normal_message() -> None:
    assert supervisor._looks_like_identity_correction(_msg("yes that's me")) is False


def test_identity_correction_empty() -> None:
    assert supervisor._looks_like_identity_correction(_msg("")) is False


# ============================================================
# _classify_patient_status
# ============================================================

def test_classify_status_first_visit() -> None:
    assert supervisor._classify_patient_status(_msg("This is my first visit")) == "new"


def test_classify_status_first_time() -> None:
    assert supervisor._classify_patient_status(_msg("First time here")) == "new"


def test_classify_status_new_patient() -> None:
    assert supervisor._classify_patient_status(_msg("I'm a new patient")) == "new"


def test_classify_status_never_been() -> None:
    assert supervisor._classify_patient_status(_msg("I've never been there")) == "new"


def test_classify_status_havent_been() -> None:
    assert supervisor._classify_patient_status(_msg("I haven't been")) == "new"


def test_classify_status_no() -> None:
    assert supervisor._classify_patient_status(_msg("No")) == "new"


def test_classify_status_nope() -> None:
    assert supervisor._classify_patient_status(_msg("Nope")) == "new"


def test_classify_status_returning() -> None:
    assert supervisor._classify_patient_status(_msg("I'm a returning patient")) == "returning"


def test_classify_status_been_there_before() -> None:
    assert supervisor._classify_patient_status(_msg("I've been there before")) == "returning"


def test_classify_status_came_before() -> None:
    assert supervisor._classify_patient_status(_msg("I came before")) == "returning"


def test_classify_status_yes() -> None:
    assert supervisor._classify_patient_status(_msg("Yes")) == "returning"


def test_classify_status_i_have() -> None:
    assert supervisor._classify_patient_status(_msg("I have")) == "returning"


def test_classify_status_ambiguous() -> None:
    assert supervisor._classify_patient_status(_msg("I'm not sure")) is None


def test_classify_status_empty() -> None:
    assert supervisor._classify_patient_status(_msg("")) is None


def test_classify_status_with_llm_returns_normalized_label(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]):
            return type("Response", (), {"content": [{"text": " Returning "}]})()

    monkeypatch.setattr(supervisor, "ChatAnthropic", FakeLLM)

    result = asyncio.run(
        supervisor._classify_patient_status_with_llm(_msg("I used to come here"))
    )

    assert result == "returning"


def test_classify_status_with_llm_returns_none_for_unknown(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]):
            return type("Response", (), {"content": "unknown"})()

    monkeypatch.setattr(supervisor, "ChatAnthropic", FakeLLM)

    result = asyncio.run(
        supervisor._classify_patient_status_with_llm(_msg("maybe"))
    )

    assert result is None


# ============================================================
# _looks_like_explicit_intent_switch
# ============================================================

def test_intent_switch_with_actually() -> None:
    assert supervisor._looks_like_explicit_intent_switch(
        _msg("Actually, I want to cancel"),
        "book",
    ) is True


def test_intent_switch_with_instead() -> None:
    assert supervisor._looks_like_explicit_intent_switch(
        _msg("I want to reschedule instead"),
        "book",
    ) is True


def test_intent_switch_with_rather() -> None:
    assert supervisor._looks_like_explicit_intent_switch(
        _msg("I'd rather just cancel it"),
        "book",
    ) is True


def test_intent_switch_marker_without_new_intent_not_detected() -> None:
    assert supervisor._looks_like_explicit_intent_switch(
        _msg("Actually, yes that sounds good"),
        "book",
    ) is False


def test_intent_switch_different_keyword() -> None:
    assert supervisor._looks_like_explicit_intent_switch(
        _msg("I want to cancel my appointment"),
        "book",
    ) is True


def test_intent_switch_same_keyword_not_detected() -> None:
    # Saying "book" when current intent is already "book" is not a switch
    assert supervisor._looks_like_explicit_intent_switch(
        _msg("book an appointment"),
        "book",
    ) is False


def test_intent_switch_no_markers_no_keyword() -> None:
    assert supervisor._looks_like_explicit_intent_switch(
        _msg("yes that sounds good"),
        "book",
    ) is False


def test_intent_switch_empty_message() -> None:
    assert supervisor._looks_like_explicit_intent_switch(
        _msg(""),
        "book",
    ) is False


def test_intent_switch_review_for_move_this_appointment() -> None:
    assert supervisor._may_need_intent_switch_review(
        _msg("I need to move this appointment"),
        "book",
    ) is True


def test_intent_switch_review_ignores_booking_detail_clarification() -> None:
    assert supervisor._may_need_intent_switch_review(
        _msg("Actually, next week works better"),
        "book",
    ) is False


def test_classify_intent_switch_with_llm_returns_normalized_label(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]):
            return type("Response", (), {"content": [{"text": " ReSchedule "}]})()

    monkeypatch.setattr(supervisor, "ChatAnthropic", FakeLLM)

    result = asyncio.run(
        supervisor._classify_intent_switch_with_llm(
            _msg("Could we move this appointment to another day?"),
            "book",
        )
    )

    assert result == "reschedule"


def test_classify_intent_switch_with_llm_returns_none_for_no_switch(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]):
            return type("Response", (), {"content": "none"})()

    monkeypatch.setattr(supervisor, "ChatAnthropic", FakeLLM)

    result = asyncio.run(
        supervisor._classify_intent_switch_with_llm(
            _msg("Actually, next week works better"),
            "book",
        )
    )

    assert result is None


# ============================================================
# _is_first_message
# ============================================================

def test_is_first_message_single_human() -> None:
    assert supervisor._is_first_message(
        _state(messages=[HumanMessage(content="hi")])
    ) is True


def test_is_first_message_with_ai_response() -> None:
    assert supervisor._is_first_message(
        _state(
            messages=[
                HumanMessage(content="hi"),
                AIMessage(content="Hello!"),
            ]
        )
    ) is False


def test_is_first_message_multiple_human() -> None:
    assert supervisor._is_first_message(
        _state(
            messages=[
                HumanMessage(content="hi"),
                AIMessage(content="Hello!"),
                HumanMessage(content="I need help"),
            ]
        )
    ) is False


def test_is_first_message_empty() -> None:
    assert supervisor._is_first_message(_state(messages=[])) is False


# ============================================================
# _latest_human_message
# ============================================================

def test_latest_human_message_returns_last() -> None:
    state = _state(
        messages=[
            HumanMessage(content="first"),
            AIMessage(content="response"),
            HumanMessage(content="second"),
        ]
    )
    result = supervisor._latest_human_message(state)
    assert result is not None
    assert result.content == "second"


def test_latest_human_message_returns_none_for_empty() -> None:
    state = _state(messages=[])
    assert supervisor._latest_human_message(state) is None


def test_latest_human_message_returns_none_for_ai_only() -> None:
    state = _state(messages=[AIMessage(content="hello")])
    assert supervisor._latest_human_message(state) is None


# ============================================================
# _flatten_message_content
# ============================================================

def test_flatten_content_string() -> None:
    assert supervisor._flatten_message_content("hello") == "hello"


def test_flatten_content_list_of_strings() -> None:
    assert supervisor._flatten_message_content(["hello", " world"]) == "hello world"


def test_flatten_content_list_of_dicts() -> None:
    result = supervisor._flatten_message_content(
        [{"text": "hello"}, {"text": " world"}]
    )
    assert result == "hello world"


def test_flatten_content_mixed_list() -> None:
    result = supervisor._flatten_message_content(
        ["hello", {"text": " world"}, {"type": "tool_use"}]
    )
    assert result == "hello world"


def test_flatten_content_none() -> None:
    assert supervisor._flatten_message_content(None) == "None"


def test_flatten_content_integer() -> None:
    assert supervisor._flatten_message_content(42) == "42"


# ============================================================
# _awaiting_patient_status_answer
# ============================================================

def test_awaiting_status_after_visit_question() -> None:
    state = _state(
        messages=[
            HumanMessage(content="I want to book"),
            AIMessage(
                content="Have you been seen at this clinic before, or is this your first visit?"
            ),
            HumanMessage(content="yes, I've been there"),
        ]
    )
    assert supervisor._awaiting_patient_status_answer(state) is True


def test_awaiting_status_after_visit_question_with_variant_punctuation() -> None:
    state = _state(
        messages=[
            HumanMessage(content="I want to book"),
            AIMessage(
                content="Have you been seen at this clinic before or is this your first visit today?"
            ),
            HumanMessage(content="yes"),
        ]
    )
    assert supervisor._awaiting_patient_status_answer(state) is True


def test_awaiting_status_after_different_question() -> None:
    state = _state(
        messages=[
            HumanMessage(content="I want to book"),
            AIMessage(content="What symptoms are you having?"),
            HumanMessage(content="headaches"),
        ]
    )
    assert supervisor._awaiting_patient_status_answer(state) is False


def test_awaiting_status_no_ai_message() -> None:
    state = _state(messages=[HumanMessage(content="hi")])
    assert supervisor._awaiting_patient_status_answer(state) is False


def test_identity_correction_llm_fallback_returns_true(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[object]):
            return type("Response", (), {"content": [{"text": " correction "}]})()

    monkeypatch.setattr(supervisor, "ChatAnthropic", FakeLLM)

    result = asyncio.run(
        supervisor._classify_identity_correction_with_llm(
            AIMessage(content="I found Sarah Connor on file — is that you?"),
            _msg("The date of birth is wrong"),
        )
    )

    assert result is True


# ============================================================
# supervisor_node — scheduling completion then new intent
# ============================================================

def test_supervisor_waits_after_scheduling_completes() -> None:
    """When scheduling clears intent (flow complete), supervisor should
    go to 'done' and wait, not ask again."""
    result = asyncio.run(
        supervisor.supervisor_node(
            _state(
                intent=None,
                last_agent="scheduling",
                patient_id="patient-1",
                patient_name="Sarah",
                messages=[
                    HumanMessage(content="hi"),
                    AIMessage(content="Your appointment is booked."),
                    HumanMessage(content="thanks"),
                ],
            )
        )
    )

    assert result["current_agent"] == "done"
    assert result.get("last_agent") is None


# ============================================================
# _latest_ai_message_before_latest_human
# ============================================================

def test_latest_ai_before_human_basic() -> None:
    state = _state(
        messages=[
            HumanMessage(content="hi"),
            AIMessage(content="first ai"),
            AIMessage(content="second ai"),
            HumanMessage(content="ok"),
        ]
    )
    result = supervisor._latest_ai_message_before_latest_human(state)
    assert result is not None
    assert result.content == "second ai"


def test_latest_ai_before_human_no_ai() -> None:
    state = _state(
        messages=[
            HumanMessage(content="hi"),
            HumanMessage(content="hello?"),
        ]
    )
    result = supervisor._latest_ai_message_before_latest_human(state)
    assert result is None


def test_latest_ai_before_human_no_human() -> None:
    state = _state(messages=[AIMessage(content="hi")])
    result = supervisor._latest_ai_message_before_latest_human(state)
    assert result is None
