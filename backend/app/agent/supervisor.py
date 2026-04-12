"""
Supervisor node for the multi-agent medical scheduling system.

The Supervisor is the central router in the graph. It:
  1. Greets the patient on first contact
  2. Classifies intent from the conversation (book/reschedule/cancel)
  3. Routes to sub-agents based on state

It does NOT have tools. It either routes to a sub-agent or
responds directly to the patient (e.g., greeting, clarifying).

Flow design:
  The Supervisor greets the patient FIRST, before any sub-agent runs.
  This is intentional — not every patient needs identification (they
  might just ask a quick question like "what's your address?"). We
  only route to Intake once we know the patient needs a service that
  requires identification (booking, rescheduling, cancelling).

Routing rules (checked in order):
  1. First message (no greeting yet)  → greet the patient
  2. Intent unknown                    → classify via LLM, or ask
  3. Intent known + no patient_id      → route to Intake Agent
  4. Intent=book + no specialty        → route to Triage Agent
  5. Intent=book + has specialty       → route to Scheduling Agent
  6. Intent=reschedule/cancel          → route to Scheduling Agent

Loop prevention:
  Each sub-agent sets ``last_agent`` when it runs. If the routing
  rules would send control to the same agent that just ran, the
  Supervisor goes to END instead — the agent is waiting for patient
  input.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import SecretStr

from app.agent.state import AgentState
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# INTENT CLASSIFIER
# ============================================================
# Used ONLY when we can't determine intent from state alone.
# This is the one place the Supervisor uses an LLM — a fast,
# constrained call that returns exactly one word.
# ============================================================

_INTENT_SYSTEM_PROMPT = """\
You are a medical clinic intent classifier. Based on the conversation, \
determine what the patient wants to do.

Respond with EXACTLY one word — no punctuation, no explanation:
- book — patient wants to book a new appointment. This includes when they \
describe symptoms or health problems (e.g., "I have headaches", "my knee hurts", \
"I've been feeling dizzy"). If someone describes a health issue, they need to see \
a doctor, which means booking.
- reschedule — patient wants to move an existing appointment to a different time
- cancel — patient wants to cancel an existing appointment
- unknown — only if there is truly no indication of what they want (e.g., "hi", \
"I have a question about parking")
"""


async def _classify_intent(
    state: AgentState,
) -> Literal["book", "reschedule", "cancel"] | None:
    """Use a fast LLM call to classify patient intent from the conversation.

    Returns one of "book", "reschedule", "cancel" if the intent is clear,
    or None if the conversation doesn't contain enough signal yet.
    """
    llm = ChatAnthropic(
        model_name=settings.anthropic_model,
        api_key=SecretStr(settings.anthropic_api_key),
        temperature=0,
        max_tokens_to_sample=10,
        timeout=None,
        stop=None,
    )

    # Only pass the last few messages to the classifier.
    # In long conversations, earlier messages about booking/triage
    # can confuse the classifier when we're trying to detect a NEW
    # intent like "reschedule" after a completed booking.
    recent_messages = state["messages"][-6:]

    response = await llm.ainvoke(
        [SystemMessage(content=_INTENT_SYSTEM_PROMPT)] + recent_messages
    )

    # response.content can be a string or a list of content blocks
    content = response.content
    if isinstance(content, list):
        # Extract text from content blocks
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    raw = content.strip().lower()

    if raw in ("book", "reschedule", "cancel"):
        return raw  # type: ignore[return-value]

    return None


def _latest_human_message(state: AgentState) -> HumanMessage | None:
    """Return the newest human message in the conversation, if any."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message
    return None


def _looks_like_identity_correction(message: HumanMessage) -> bool:
    """Detect when the patient is correcting the UIN we have on file."""
    normalized = " ".join(str(message.content).lower().split())
    correction_markers = (
        "my uin is",
        "uin is",
        "wrong uin",
        "different uin",
        "wrong id",
        "different id",
    )
    return any(marker in normalized for marker in correction_markers)


# ============================================================
# HELPERS
# ============================================================

def _is_first_message(state: AgentState) -> bool:
    """Check if this is the very first message in the conversation.

    True when the only message is a single human message — the
    patient just said "hi" or similar and nobody has responded yet.
    """
    messages = state.get("messages", [])

    # Count human messages — if there's only one and no AI messages,
    # this is the first turn
    human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
    ai_count = sum(1 for m in messages if isinstance(m, AIMessage))

    return human_count == 1 and ai_count == 0


# ============================================================
# SUPERVISOR NODE
# ============================================================

async def supervisor_node(state: AgentState) -> dict:
    """Decide which sub-agent should handle the next turn.

    Returns a state update. The key field is ``current_agent``,
    which the graph's conditional edges read to route control.

    ``current_agent`` values:
      - "intake" / "triage" / "scheduling" — route to that sub-agent
      - "supervisor" — re-run Supervisor (after setting a new field
        like intent, so routing rules can re-evaluate)
      - "done" — Supervisor responded to the patient directly,
        OR a sub-agent is waiting for patient input;
        go to END and wait for the next human message
    """

    # ── Read state with safe defaults ──────────────────
    patient_id = state.get("patient_id")
    patient_name = state.get("patient_name")
    intent = state.get("intent")
    specialty_id = state.get("specialty_id")
    last_agent = state.get("last_agent")

    # ── Rule 1: Greet on first message ───────────────────
    # The patient just connected. Greet them naturally and
    # let them tell us what they need — don't jump straight
    # to asking for their UIN.
    if _is_first_message(state):
        logger.info("First message — greeting patient")
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Hello! Welcome to the university health clinic. "
                        "How can I help you today?"
                    )
                )
            ],
            "current_agent": "done",
        }

    # ── Rule 2: Determine intent if unknown ──────────────
    if intent is None:
        # If scheduling just completed (it reset intent and asked
        # "anything else?"), don't pile on another question —
        # just wait for the patient's response.
        if last_agent == "scheduling":
            logger.info("Scheduling just completed — waiting for patient input")
            return {"current_agent": "done", "last_agent": None}

        classified = await _classify_intent(state)

        if classified is not None:
            logger.info(f"Intent classified as: {classified}")
            # Re-run Supervisor with the new intent so routing
            # rules can evaluate it on the next pass. Clear any
            # stale appointment selection from the last finished flow.
            return {
                "intent": classified,
                "selected_appointment_id": None,
                "current_agent": "supervisor",
                "last_agent": None,
            }

        # Still unclear — ask the patient to clarify.
        # Keep the tone neutral — we don't know if the patient is
        # describing something serious or just saying hello.
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Of course! Are you looking to book a new "
                        "appointment, reschedule an existing one, "
                        "or cancel?"
                    )
                )
            ],
            "current_agent": "done",
        }

    # ── Rule 2b: Allow explicit mid-conversation intent switches ─────
    # Patients often change course with phrases like "actually I'd like
    # to reschedule instead." If we keep the old intent, the wrong
    # sub-agent will answer from stale context.
    latest_human = _latest_human_message(state)
    if latest_human is not None:
        override_intent = await _classify_intent(
            {
                "messages": [latest_human],
                "patient_id": patient_id,
                "patient_name": patient_name,
                "symptoms": [],
                "specialty_id": None,
                "appointment_id": state.get("appointment_id"),
                "selected_appointment_id": state.get("selected_appointment_id"),
                "current_agent": "supervisor",
                "intent": intent,
                "last_agent": last_agent,
            }
        )

        if override_intent is not None and override_intent != intent:
            logger.info(f"Intent changed mid-flow: {intent} -> {override_intent}")
            return {
                "intent": override_intent,
                "symptoms": [],
                "specialty_id": None,
                "selected_appointment_id": None,
                "current_agent": "supervisor",
                "last_agent": None,
            }

    # ── Rule 2c: Let the patient correct a mistaken UIN mid-flow ───────
    if (
        latest_human is not None
        and patient_id is not None
        and _looks_like_identity_correction(latest_human)
    ):
        logger.info("Patient corrected UIN mid-conversation — returning to intake")
        return {
            "patient_id": None,
            "patient_name": None,
            "appointment_id": None,
            "selected_appointment_id": None,
            "current_agent": "intake",
            "last_agent": None,
        }

    # ── Rule 3: Need identification for any action ───────
    # Now we know the intent requires a service (book/reschedule/
    # cancel), so we need to identify the patient first.
    if patient_id is None:
        next_agent = "intake"
        if next_agent == last_agent:
            logger.info("Intake already ran — waiting for patient input")
            return {"current_agent": "done", "last_agent": None}
        return {"current_agent": "intake"}

    # ── Rule 4: Booking flow — need triage first? ────────
    if intent == "book":
        next_agent = "triage" if specialty_id is None else "scheduling"
        if next_agent == last_agent:
            logger.info(f"Agent '{next_agent}' already ran — waiting for patient input")
            return {"current_agent": "done", "last_agent": None}
        return {"current_agent": next_agent}

    # ── Rule 5: Reschedule or cancel → scheduling ────────
    if intent in ("reschedule", "cancel"):
        if "scheduling" == last_agent:
            logger.info("Scheduling already ran — waiting for patient input")
            return {"current_agent": "done", "last_agent": None}
        return {"current_agent": "scheduling"}

    # ── Fallback: nothing left to do ─────────────────────
    return {"current_agent": "done"}
