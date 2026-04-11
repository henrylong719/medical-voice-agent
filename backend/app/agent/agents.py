# backend/app/agent/agents.py

"""
Sub-agent definitions for the multi-agent medical scheduling system.

Each sub-agent is a focused mini-agent with:
  - Its own system prompt (short, role-specific)
  - Its own tool subset (enforces role boundaries)
  - A node function that runs the agent and updates shared state

Sub-agents are built with create_agent, which gives each one
its own internal observe → think → act loop. They can call tools
multiple times before returning control to the Supervisor.

No checkpointer on sub-agents — the outer graph owns persistence.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agent.state import AgentState
from app.agent.tools import (
    identify_patient,
    register_patient,
    triage_symptoms,
    list_specialties,
    find_slots,
    book_appointment,
    find_appointment,
    reschedule_appointment,
    cancel_appointment,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# SHARED LLM BUILDER
# ============================================================

def _build_llm() -> ChatAnthropic:
    """Build a Claude instance shared by all sub-agents."""
    return ChatAnthropic(
        model_name=settings.anthropic_model,
        api_key=SecretStr(settings.anthropic_api_key),
        temperature=0,
        max_tokens_to_sample=1024,
        timeout=None,
        stop=None,
    )


# ============================================================
# SYSTEM PROMPTS
# ============================================================
# Each prompt is short and focused. No conflicting instructions,
# no knowledge of other agents' responsibilities.
# ============================================================

_INTAKE_PROMPT = """\
You are the intake assistant at a university medical clinic. Your ONLY job \
is to identify the patient.

## What to do
1. Greet the patient warmly and ask for their 9-digit UIN (university ID number).
2. Once they provide it, call identify_patient to look them up.
3. If found — confirm their name ("Welcome back, Sarah!") and STOP. \
Do not ask what they need. Do not continue the conversation. Just confirm and stop.
4. If not found — let them know, ask for their full name (and optionally phone number), \
then call register_patient. After registration, confirm and STOP.

## Rules
- Always ask for UIN first. Do not ask for their name before trying the UIN.
- If the patient gives something that is not a 9-digit number, politely ask again.
- Keep responses to 1-2 sentences.
- Do NOT ask what brings them in or discuss symptoms — that is not your job.
"""

_TRIAGE_PROMPT = """\
You are the triage assistant at a university medical clinic. Your ONLY job \
is to understand the patient's symptoms and match them to the right specialty.

## What to do
1. Ask the patient to describe what they're experiencing.
2. Listen carefully. If their description is vague, ask ONE focused follow-up \
question to clarify (e.g., "Where exactly is the pain?" or "How long has this \
been going on?").
3. Once you have enough detail, call triage_symptoms with both the individual \
symptoms list AND the patient's full natural language description.
4. Review the triage results. If a clear specialty match exists, confirm it with \
the patient ("Based on what you're describing, I'd recommend seeing a cardiologist. \
Does that sound right?") and STOP.
5. If results are ambiguous, ask the follow-up questions from the triage results \
to narrow it down.

## Rules
- Do NOT give medical advice, diagnoses, or suggest treatments.
- Do NOT discuss medications or dosages.
- If the patient asks for medical advice, say: "I can help you get an appointment \
with a specialist who can help with that."
- Keep responses to 1-3 sentences.
- If the patient describes emergency symptoms (severe chest pain + arm/jaw pain, \
difficulty breathing, signs of stroke), IMMEDIATELY tell them to call 911 or go \
to the nearest ER. Do NOT attempt triage.
"""

_SCHEDULING_PROMPT = """\
You are the scheduling assistant at a university medical clinic. Your ONLY job \
is to help the patient book, reschedule, or cancel appointments.

## For new bookings
1. Ask if the patient has a day or time preference ("Do you have a preferred day \
or time of day?").
2. Call find_slots with the specialty and their preferences.
3. Present 2-3 options clearly, with doctor name, day, and time.
4. Once the patient picks a slot, confirm ALL details before booking — doctor name, \
specialty, full date (day of week + month + day number), and time. \
Example: "I'll book you with Dr. Smith for cardiology on Monday, April 13th at 2 PM. \
Shall I confirm?"
5. Only call book_appointment AFTER the patient explicitly confirms.

## For reschedules
1. Call find_appointment to find their existing appointments.
2. Confirm which appointment they want to reschedule.
3. Call reschedule_appointment — this cancels the old one and finds new slots.
4. Present new options and book after confirmation.

## For cancellations
1. Call find_appointment to find their existing appointments.
2. Confirm which appointment they want to cancel.
3. Only call cancel_appointment AFTER the patient explicitly confirms.

## Rules
- ALWAYS ask for explicit confirmation before booking, rescheduling, or cancelling.
- ALWAYS include the full date (day + month + day number) when presenting or confirming \
appointment times. Never say just "Monday" — say "Monday, April 13th".
- Do NOT fabricate appointment times or doctor names — only use data from tool results.
- Do NOT give medical advice.
- Keep responses to 1-3 sentences, except when listing appointment options.
"""


# ============================================================
# AGENT CACHE
# ============================================================
# create_agent builds a compiled graph internally, so we cache
# to avoid rebuilding on every turn. No checkpointer here —
# the outer multi-agent graph handles persistence.
# ============================================================

_intake_agent: Any | None = None
_triage_agent: Any | None = None
_scheduling_agent: Any | None = None


def _get_intake_agent() -> Any:
    """Build or return cached Intake Agent."""
    global _intake_agent
    if _intake_agent is None:
        _intake_agent = create_agent(
            model=_build_llm(),
            tools=[identify_patient, register_patient],
            system_prompt=_INTAKE_PROMPT,
        )
    return _intake_agent


def _get_triage_agent() -> Any:
    """Build or return cached Triage Agent."""
    global _triage_agent
    if _triage_agent is None:
        _triage_agent = create_agent(
            model=_build_llm(),
            tools=[triage_symptoms, list_specialties],
            system_prompt=_TRIAGE_PROMPT,
        )
    return _triage_agent


def _get_scheduling_agent() -> Any:
    """Build or return cached Scheduling Agent."""
    global _scheduling_agent
    if _scheduling_agent is None:
        _scheduling_agent = create_agent(
            model=_build_llm(),
            tools=[
                find_slots,
                book_appointment,
                find_appointment,
                reschedule_appointment,
                cancel_appointment,
            ],
            system_prompt=_SCHEDULING_PROMPT,
        )
    return _scheduling_agent


# ============================================================
# NODE FUNCTIONS
# ============================================================
# Each node function:
#   1. Gets the cached sub-agent
#   2. Runs it with the current conversation messages
#   3. Scans tool messages to extract state updates
#   4. Returns new messages + updated state fields
#
# Only NEW messages are returned (sliced from input_count),
# because the add_messages reducer appends them to the
# existing history — returning all would create duplicates.
# ============================================================

async def intake_node(state: AgentState) -> dict:
    """Run the Intake Agent and extract patient identity from results.

    After the agent finishes (it identified or registered a patient),
    we scan the tool messages to find the patient_id and name so we
    can set them in the shared state for other agents to use.
    """
    agent = _get_intake_agent()
    result = await agent.ainvoke({"messages": state["messages"]})
    new_messages = result["messages"]

    # ── Extract patient identity from tool results ───────
    patient_id = state["patient_id"]
    patient_name = state["patient_name"]

    for msg in new_messages:
        if not hasattr(msg, "type") or msg.type != "tool":
            continue

        content = msg.content
        if "Patient found:" in content or "Successfully registered" in content:
            # Extract ID from "ID: abc-123-..."
            if "ID: " in content:
                id_start = content.index("ID: ") + 4
                id_end = content.index(",", id_start)
                patient_id = content[id_start:id_end].strip()

            # Extract name from "Patient found: Sarah (" or
            # "Successfully registered Sarah ("
            for prefix in ("Patient found: ", "Successfully registered "):
                if prefix in content:
                    name_start = content.index(prefix) + len(prefix)
                    name_end = content.index(" (", name_start)
                    patient_name = content[name_start:name_end].strip()
                    break

    input_count = len(state["messages"])

    return {
        "messages": new_messages[input_count:],
        "patient_id": patient_id,
        "patient_name": patient_name,
    }


async def triage_node(state: AgentState) -> dict:
    """Run the Triage Agent and extract specialty match from results.

    After the agent collects symptoms and runs triage_symptoms,
    we look for a specialty ID in the tool results (the top-ranked
    match) and extract the symptom list from the tool call arguments.
    """
    agent = _get_triage_agent()
    result = await agent.ainvoke({"messages": state["messages"]})
    new_messages = result["messages"]

    specialty_id = state["specialty_id"]
    symptoms = list(state["symptoms"])

    for msg in new_messages:
        if not hasattr(msg, "type"):
            continue

        # ── Extract specialty from triage tool results ───
        # The triage tool returns lines like:
        #   "- Cardiology (ID: abc-123): score 8.50, matched on: chest pain"
        # We take the first match as the primary recommendation.
        if msg.type == "tool" and "(ID: " in msg.content:
            lines = msg.content.split("\n")
            for line in lines:
                if "(ID: " in line and ("score" in line or "similarity" in line):
                    id_start = line.index("(ID: ") + 5
                    id_end = line.index(")", id_start)
                    specialty_id = line[id_start:id_end].strip()
                    break

        # ── Extract symptoms from tool call arguments ────
        # When the LLM calls triage_symptoms, LangChain stores
        # the tool call args in the AIMessage. We read them to
        # get the structured symptom list the LLM extracted
        # from the patient's description.
        if msg.type == "ai" and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc["name"] == "triage_symptoms":
                    args = tc.get("args", {})
                    if "symptoms" in args:
                        symptoms = args["symptoms"]

    input_count = len(state["messages"])

    return {
        "messages": new_messages[input_count:],
        "specialty_id": specialty_id,
        "symptoms": symptoms,
    }


async def scheduling_node(state: AgentState) -> dict:
    """Run the Scheduling Agent and extract appointment ID from results.

    The Scheduling Agent handles booking, rescheduling, and cancelling.
    After it runs, we check for a newly created appointment ID in the
    tool results (from book_appointment's response).
    """
    agent = _get_scheduling_agent()
    result = await agent.ainvoke({"messages": state["messages"]})
    new_messages = result["messages"]

    appointment_id = state["appointment_id"]

    for msg in new_messages:
        if not hasattr(msg, "type") or msg.type != "tool":
            continue

        # ── Extract appointment ID from booking result ───
        # book_appointment returns: "Appointment ID: abc-123."
        if "Appointment ID: " in msg.content:
            id_start = msg.content.index("Appointment ID: ") + 16
            id_end = msg.content.index(".", id_start)
            appointment_id = msg.content[id_start:id_end].strip()

    input_count = len(state["messages"])

    return {
        "messages": new_messages[input_count:],
        "appointment_id": appointment_id,
    }