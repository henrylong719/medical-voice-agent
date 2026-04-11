# backend/app/agent/supervisor.py

"""
Supervisor node for the multi-agent medical scheduling system.

The Supervisor is the central router in the graph. It:
  1. Checks structured state fields (patient_id, intent, specialty_id)
  2. Uses rule-based logic to decide which sub-agent runs next
  3. Falls back to a quick LLM call only when intent is unclear

It does NOT have tools. It either routes to a sub-agent or
responds directly to the patient (e.g., asking clarifying questions).
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import SecretStr

from app.agent.state import AgentState
from app.config import settings

logger = logging.getLogger(__name__)

# ── Intent classifier prompt ────────────────────────────────
# Used ONLY when we can't determine intent from state alone.
# Constrained to a single-word response for reliability.

_INTENT_SYSTEM_PROMPT = """\
You are a medical clinic intent classifier. Based on the conversation, \
determine what the patient wants to do.

Respond with EXACTLY one word — no punctuation, no explanation:
- book — patient wants to book a new appointment (including symptom descriptions)
- reschedule — patient wants to move an existing appointment to a different time
- cancel — patient wants to cancel an existing appointment
- unknown — not enough information yet
"""


def _extract_text_content(content: object) -> str:
    """Normalize model output content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

        return "".join(text_parts)

    return ""


async def _classify_intent(
    state: AgentState,
) -> Literal["book", "reschedule", "cancel"] | None:
    """Use a fast LLM call to classify patient intent from the conversation."""

    llm = ChatAnthropic(
        model_name=settings.anthropic_model,
        api_key=SecretStr(settings.anthropic_api_key),
        temperature=0,
        max_tokens_to_sample=10,  # We only need one word
        timeout=None,
        stop=None,
    )

    # Pass the conversation so far + the classification prompt
    response = await llm.ainvoke(
        [SystemMessage(content=_INTENT_SYSTEM_PROMPT)] + state["messages"]
    )

    raw = _extract_text_content(response.content).strip().lower()

    if raw == "book":
        return "book"
    if raw == "reschedule":
        return "reschedule"
    if raw == "cancel":
        return "cancel"

    return None


# ── Supervisor routing logic ────────────────────────────────

async def supervisor_node(state: AgentState) -> dict:
    """Decide which sub-agent should handle the next turn.

    Returns a state update. The key field is `current_agent`,
    which the graph's conditional edges read to route control.

    Routing rules (checked in order):
      1. No patient_id       → intake
      2. No intent            → classify, or ask patient
      3. intent=book, no specialty → triage
      4. intent=book, has specialty → scheduling
      5. intent=reschedule/cancel   → scheduling
    """

    # ── Rule 1: Patient must be identified first ─────────
    if state["patient_id"] is None:
        return {"current_agent": "intake"}

    # ── Rule 2: Determine intent if unknown ──────────────
    if state["intent"] is None:
        classified = await _classify_intent(state)

        if classified is not None:
            logger.info(f"Intent classified as: {classified}")
            return {
                "intent": classified,
                "current_agent": "supervisor",  # Re-run supervisor with new intent
            }

        # Still unclear — ask the patient directly
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Hi {state['patient_name'] or 'there'}! "
                        "How can I help you today — are you looking to "
                        "book a new appointment, reschedule an existing one, "
                        "or cancel?"
                    )
                )
            ],
            "current_agent": "supervisor",  # Will re-run when patient replies
        }

    # ── Rule 3: Booking flow — need triage first? ────────
    if state["intent"] == "book":
        if state["specialty_id"] is None:
            return {"current_agent": "triage"}
        return {"current_agent": "scheduling"}

    # ── Rule 4: Reschedule or cancel — straight to scheduling
    if state["intent"] in ("reschedule", "cancel"):
        return {"current_agent": "scheduling"}

    # Fallback (shouldn't reach here)
    return {"current_agent": "intake"}
