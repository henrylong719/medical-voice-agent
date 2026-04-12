from __future__ import annotations

import pytest

from evals import eval_helpers
from evals.eval_helpers import eval_report, parse_judge_json


def test_parse_judge_json_accepts_plain_json() -> None:
    payload = parse_judge_json('{"passed": true, "notes": "ok"}')

    assert payload == {"passed": True, "notes": "ok"}


def test_parse_judge_json_accepts_code_fences_and_trailing_text() -> None:
    payload = parse_judge_json(
        """```json
{"passed": true, "notes": "ok"}
```
Extra commentary that should be ignored.
"""
    )

    assert payload == {"passed": True, "notes": "ok"}


@pytest.mark.asyncio
async def test_run_conversation_calls_after_agent_turn_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            "I'll book you for Monday, April 20th at 9:00 AM. Shall I confirm?",
            "Goodbye!",
        ]
    )

    async def fake_invoke_agent(_message: str, *, thread_id: str) -> str:
        assert thread_id.startswith("eval-")
        return next(replies)

    observed_agent_turns: list[str] = []

    async def after_agent_turn(history: list[dict]) -> None:
        observed_agent_turns.append(history[-1]["content"])

    monkeypatch.setattr(eval_helpers, "invoke_agent", fake_invoke_agent)
    monkeypatch.setattr(eval_helpers, "simulate_user_turn", lambda *_: "Yes")

    history = await eval_helpers.run_conversation(
        "Hi",
        "persona",
        max_turns=2,
        after_agent_turn=after_agent_turn,
        stop_phrases=("goodbye",),
    )

    assert observed_agent_turns == [
        "I'll book you for Monday, April 20th at 9:00 AM. Shall I confirm?",
        "Goodbye!",
    ]
    assert history == [
        {"role": "user", "content": "Hi"},
        {
            "role": "agent",
            "content": "I'll book you for Monday, April 20th at 9:00 AM. Shall I confirm?",
        },
        {"role": "user", "content": "Yes"},
        {"role": "agent", "content": "Goodbye!"},
    ]


def test_eval_report_prints_full_transcript_for_quality_failures(capsys: pytest.CaptureFixture[str]) -> None:
    safety_rate, quality_rate = eval_report(
        [
            {
                "run": 0,
                "safe": True,
                "judgment": {
                    "quality_ok": False,
                    "notes": "missed the confirmation step",
                },
                "transcript": [
                    {"role": "user", "content": "I need help"},
                    {"role": "agent", "content": "Sure, what do you need?"},
                ],
            }
        ],
        1,
        safety_keys=("safe",),
        quality_keys=("quality_ok",),
    )

    output = capsys.readouterr().out

    assert safety_rate == 1.0
    assert quality_rate == 0.0
    assert "--- FAILED RUN 0 TRANSCRIPT ---" in output
    assert "failed quality checks: quality_ok" in output
    assert "USER: I need help" in output
    assert "AGENT: Sure, what do you need?" in output


def test_eval_report_stays_quiet_when_lenient_threshold_still_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    safety_rate, quality_rate = eval_report(
        [
            {
                "run": 0,
                "safe": True,
                "judgment": {
                    "quality_ok": False,
                    "notes": "one run missed",
                },
                "transcript": [
                    {"role": "user", "content": "hello"},
                    {"role": "agent", "content": "hi"},
                ],
            },
            {
                "run": 1,
                "safe": True,
                "judgment": {
                    "quality_ok": True,
                    "notes": "ok",
                },
                "transcript": [
                    {"role": "user", "content": "hello again"},
                    {"role": "agent", "content": "hi again"},
                ],
            },
            {
                "run": 2,
                "safe": True,
                "judgment": {
                    "quality_ok": True,
                    "notes": "ok",
                },
                "transcript": [
                    {"role": "user", "content": "third"},
                    {"role": "agent", "content": "response"},
                ],
            },
        ],
        3,
        safety_keys=("safe",),
        quality_keys=("quality_ok",),
        min_quality_rate=0.6,
    )

    output = capsys.readouterr().out

    assert safety_rate == 1.0
    assert quality_rate == pytest.approx(2 / 3)
    assert "--- FAILED RUN" not in output
