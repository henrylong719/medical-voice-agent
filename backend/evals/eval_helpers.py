"""
Shared infrastructure for eval tests.

Provides:
  - simulate_user_turn(): LLM-driven patient simulator
  - run_conversation(): multi-turn driver that alternates user/agent
  - judge_transcript(): LLM judge for conversational quality
  - SeededPatient: dataclass for test fixtures
  - cleanup helpers
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable
import json
import uuid
from dataclasses import dataclass
from json import JSONDecodeError

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from app.core.config import settings
from app.agent.graph import invoke_agent
from app.supabase_client import supabase

MAX_TURNS = 14
ConversationHook = Callable[[list[dict]], Awaitable[None] | None]


@dataclass(frozen=True)
class SeededPatient:
    id: str
    full_name: str
    date_of_birth: str
    phone: str


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------


def build_eval_llm(*, model: str, max_tokens: int) -> ChatAnthropic:
    """Build an eval LLM using the app's configured Anthropic key."""
    api_key = settings.ANTHROPIC_API_KEY.strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured in backend/.env, so evals cannot run."
        )

    return ChatAnthropic(
        model_name=settings.ANTHROPIC_MODEL,
        api_key=SecretStr(settings.ANTHROPIC_API_KEY),
        temperature=0,
        max_tokens_to_sample=1024,
        timeout=None,
        stop=None,
    )


def simulate_user_turn(
    persona_prompt: str,
    history: list[dict],
) -> str:
    """Drive one user turn using an LLM with the given persona."""
    llm = build_eval_llm(model="claude-sonnet-4-6", max_tokens=200)
    messages: list = [SystemMessage(content=persona_prompt)]
    for turn in history:
        if turn["role"] == "agent":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    result = llm.invoke(messages)
    return result.content if isinstance(result.content, str) else str(result.content)


# ---------------------------------------------------------------------------
# Conversation driver
# ---------------------------------------------------------------------------


async def run_conversation(
    opening: str,
    persona_prompt: str,
    *,
    max_turns: int = MAX_TURNS,
    after_agent_turn: ConversationHook | None = None,
    stop_phrases: tuple[str, ...] = (
        "anything else",
        "have a good",
        "goodbye",
        "can i help with anything",
    ),
) -> list[dict]:
    """Run a full multi-turn conversation and return the transcript."""
    thread_id = f"eval-{uuid.uuid4()}"
    history: list[dict] = []

    user_msg = opening
    for _ in range(max_turns):
        history.append({"role": "user", "content": user_msg})
        agent_reply = await invoke_agent(user_msg, thread_id=thread_id)
        history.append({"role": "agent", "content": agent_reply})

        if after_agent_turn is not None:
            hook_result = after_agent_turn(history)
            if isawaitable(hook_result):
                await hook_result

        lowered = agent_reply.lower()
        if any(s in lowered for s in stop_phrases):
            break

        user_msg = simulate_user_turn(persona_prompt, history)

    return history


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


def judge_transcript(rubric_prompt: str, history: list[dict]) -> dict:
    """Score a transcript against a rubric using an LLM judge."""
    transcript = "\n".join(
        f"{turn['role'].upper()}: {turn['content']}" for turn in history
    )
    llm = build_eval_llm(model="claude-opus-4-6", max_tokens=600)
    result = llm.invoke([HumanMessage(content=rubric_prompt + transcript)])
    raw = result.content if isinstance(result.content, str) else str(result.content)
    return parse_judge_json(raw)


def parse_judge_json(raw: str) -> dict:
    """Parse judge output, tolerating code fences or trailing commentary."""
    cleaned = raw.strip()
    cleaned = (
        cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    )

    try:
        return json.loads(cleaned)
    except JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    first_brace = cleaned.find("{")
    while first_brace != -1:
        try:
            parsed, _end = decoder.raw_decode(cleaned[first_brace:])
        except JSONDecodeError:
            first_brace = cleaned.find("{", first_brace + 1)
            continue
        if isinstance(parsed, dict):
            return parsed
        first_brace = cleaned.find("{", first_brace + 1)

    raise JSONDecodeError("Judge output did not contain valid JSON", cleaned, 0)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def cleanup_by_tag(tag: str) -> None:
    """Delete all rows tagged with this eval scenario."""
    supabase.table("appointments").delete().eq("eval_tag", tag).execute()
    supabase.table("patients").delete().eq("eval_tag", tag).execute()


def seed_patient(patient: SeededPatient, tag: str) -> None:
    """Upsert a patient row tagged for eval cleanup."""
    supabase.table("patients").upsert(
        {
            "id": patient.id,
            "full_name": patient.full_name,
            "date_of_birth": patient.date_of_birth,
            "phone": patient.phone,
            "eval_tag": tag,
        }
    ).execute()


def count_appointments_for_patient(patient_id: str) -> int:
    """Count all appointment rows for a patient."""
    result = (
        supabase.table("appointments")
        .select("id")
        .eq("patient_id", patient_id)
        .execute()
    )
    return len(result.data or [])


def get_appointment_status(appointment_id: str) -> str | None:
    """Get the current status of an appointment."""
    result = (
        supabase.table("appointments")
        .select("status")
        .eq("id", appointment_id)
        .execute()
    )
    rows = result.data or []
    return rows[0]["status"] if rows else None


def _failed_result_keys(
    result: dict, keys: tuple[str, ...], *, from_judgment: bool
) -> list[str]:
    """Return the failed boolean keys from either the result dict or its judgment."""
    source = result.get("judgment", {}) if from_judgment else result
    if not isinstance(source, dict):
        return list(keys)
    return [key for key in keys if not bool(source.get(key, False))]


def print_failed_transcript(
    run_idx: int,
    history: list[dict],
    judgment: dict,
    *,
    failed_safety_keys: list[str] | None = None,
    failed_quality_keys: list[str] | None = None,
) -> None:
    """Print a failed run's transcript for debugging."""
    print(f"\n--- FAILED RUN {run_idx} TRANSCRIPT ---")
    if failed_safety_keys:
        print(f"  failed safety checks: {', '.join(failed_safety_keys)}")
    if failed_quality_keys:
        print(f"  failed quality checks: {', '.join(failed_quality_keys)}")
    for turn in history:
        print(f"  {turn['role'].upper()}: {turn['content']}")
    print(f"  judgment: {judgment}")


def eval_report(
    results: list[dict],
    n_runs: int,
    *,
    safety_keys: tuple[str, ...],
    quality_keys: tuple[str, ...],
    min_safety_rate: float = 1.0,
    min_quality_rate: float = 0.8,
) -> tuple[float, float]:
    """Compute and print safety/quality pass rates. Returns (safety_rate, quality_rate)."""
    safety_passes = sum(1 for r in results if all(r[k] for k in safety_keys))
    safety_rate = safety_passes / n_runs

    quality_passes = sum(
        1 for r in results if all(r["judgment"].get(k, False) for k in quality_keys)
    )
    quality_rate = quality_passes / n_runs

    print(f"\nSafety pass rate:  {safety_rate:.0%} ({safety_passes}/{n_runs})")
    print(f"Quality pass rate: {quality_rate:.0%} ({quality_passes}/{n_runs})")

    report_failed = safety_rate < min_safety_rate or quality_rate < min_quality_rate
    if report_failed:
        for r in results:
            failed_safety_keys = _failed_result_keys(
                r,
                safety_keys,
                from_judgment=False,
            )
            failed_quality_keys = _failed_result_keys(
                r,
                quality_keys,
                from_judgment=True,
            )
            if not failed_safety_keys and not failed_quality_keys:
                continue
            print_failed_transcript(
                r["run"],
                r["transcript"],
                r["judgment"],
                failed_safety_keys=failed_safety_keys,
                failed_quality_keys=failed_quality_keys,
            )

    return safety_rate, quality_rate
