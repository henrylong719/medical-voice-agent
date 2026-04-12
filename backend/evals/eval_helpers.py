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

import json
import uuid
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.graph import invoke_agent
from app.supabase_client import supabase

MAX_TURNS = 14


@dataclass(frozen=True)
class SeededPatient:
    id: str
    full_name: str
    date_of_birth: str
    phone: str


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------

def simulate_user_turn(
    persona_prompt: str,
    history: list[dict],
) -> str:
    """Drive one user turn using an LLM with the given persona."""
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=200)
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
    llm = ChatAnthropic(model="claude-opus-4-6", max_tokens=600)
    result = llm.invoke([HumanMessage(content=rubric_prompt + transcript)])
    raw = result.content if isinstance(result.content, str) else str(result.content)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def cleanup_by_tag(tag: str) -> None:
    """Delete all rows tagged with this eval scenario."""
    supabase.table("appointments").delete().eq("eval_tag", tag).execute()
    supabase.table("patients").delete().eq("eval_tag", tag).execute()


def seed_patient(patient: SeededPatient, tag: str) -> None:
    """Upsert a patient row tagged for eval cleanup."""
    supabase.table("patients").upsert({
        "id": patient.id,
        "full_name": patient.full_name,
        "date_of_birth": patient.date_of_birth,
        "phone": patient.phone,
        "eval_tag": tag,
    }).execute()


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


def print_failed_transcript(run_idx: int, history: list[dict], judgment: dict) -> None:
    """Print a failed run's transcript for debugging."""
    print(f"\n--- FAILED RUN {run_idx} TRANSCRIPT ---")
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
    safety_passes = sum(
        1 for r in results
        if all(r[k] for k in safety_keys)
    )
    safety_rate = safety_passes / n_runs

    quality_passes = sum(
        1 for r in results
        if all(r["judgment"].get(k, False) for k in quality_keys)
    )
    quality_rate = quality_passes / n_runs

    print(f"\nSafety pass rate:  {safety_rate:.0%} ({safety_passes}/{n_runs})")
    print(f"Quality pass rate: {quality_rate:.0%} ({quality_passes}/{n_runs})")

    for r in results:
        if not all(r[k] for k in safety_keys):
            print_failed_transcript(r["run"], r["transcript"], r["judgment"])

    return safety_rate, quality_rate
