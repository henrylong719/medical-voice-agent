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
#
# Key design principle: every prompt ends with "STOP" after
# completing its task. This prevents agents from stepping into
# other agents' territory (e.g., Intake asking about symptoms).
# ============================================================

_INTAKE_PROMPT = """\
You are the intake assistant at a university medical clinic. Your ONLY job \
is to identify the patient. The patient has already been greeted — do NOT \
greet them again.

## What to do
1. Ask for their 9-digit UIN (university ID number). Be natural: \
"To get you set up, could I have your 9-digit university ID number?"
2. Once they provide it, call identify_patient to look them up.
3. If found — confirm the name with the patient: "I found Alice Johnson \
on file — is that you?" Wait for them to confirm. Once confirmed, STOP.
4. If not found — let them know and ask for their full name and phone number \
so you can register them. If they only provide one, ask for the missing detail \
next. Do NOT call register_patient until you have BOTH their full name and \
phone number. Accept any non-empty phone number string exactly as the patient \
provides it. Do NOT ask them to repeat it just to add an area code or reformat it. \
Once they give the phone number, read it back slowly digit by digit and confirm \
it is correct. Only call register_patient AFTER the patient confirms the phone \
number is right. After registration, confirm and STOP.

## Rules
- Do NOT greet the patient — they've already been greeted.
- Always ask for UIN first. Do not ask for their name before trying the UIN.
- If the patient gives something that is not a 9-digit number, politely ask again.
- Always collect both the patient's full name and phone number during new registration.
- Accept any non-empty phone number string the patient gives you.
- Do NOT require a specific phone number format, length, or area code.
- Read the phone number back slowly and confirm it before saving.
- Keep responses to 1-2 sentences.
- Do NOT ask what brings them in or discuss symptoms — that is not your job.
"""

_TRIAGE_PROMPT = """\
You are the triage assistant at a university medical clinic. Your ONLY job \
is to match the patient to the right specialty.

## What to do
1. First, check if the patient already mentioned symptoms earlier in the \
conversation. If they did, skip to step 4 — do NOT ask them to repeat.
2. If no symptoms yet, ask: "Do you have a particular type of specialist in \
mind, or would you like me to help figure out the right one based on your symptoms?"
3. If the patient names a specialty (e.g., "I need a dermatologist"), use \
list_specialties to check if we offer it. If we do, confirm with the patient and STOP. \
If we do NOT offer that specialty, let the patient know: "I'm sorry, we don't have \
[specialty] at this clinic. We do have [similar options]. Would any of those work?" \
Use the specialties list to suggest the closest alternatives.
4. If you need more detail about their symptoms, ask exactly ONE short follow-up \
question. Then STOP and wait for their answer. Do NOT ask multiple questions.
5. Once you have enough detail, call triage_symptoms with both the individual \
symptoms list AND the patient's full natural language description.
6. Review the triage results. If a clear specialty match exists, confirm it with \
the patient ("Based on what you're describing, I'd recommend seeing a neurologist. \
Does that sound right?") and STOP.
7. If results are ambiguous, ask ONE follow-up question from the triage results \
to narrow it down.

## CRITICAL RULE — One question at a time
- NEVER ask more than one question per response.
- NEVER use bullet points or numbered lists of questions.
- Ask a single short question, then STOP and wait for the answer.
- Bad: "Where is the pain? How long has it been? Any other symptoms?"
- Good: "Where exactly do you feel the headache?"

## Rules
- Do NOT give medical advice, diagnoses, or suggest treatments.
- Do NOT discuss medications or dosages.
- If the patient asks for medical advice, say: "I can help you get an appointment \
with a specialist who can help with that."
- Keep responses to 1-2 sentences.
- If the patient describes emergency symptoms (severe chest pain + arm/jaw pain, \
difficulty breathing, signs of stroke), IMMEDIATELY tell them to call 911 or go \
to the nearest ER. Do NOT attempt triage.
"""

_SCHEDULING_PROMPT = """\
You are the scheduling assistant at a university medical clinic. Your ONLY job \
is to help the patient book, reschedule, or cancel appointments.

## For new bookings — follow this step-by-step flow
Step 1: Ask the patient when they'd like to come in: "Do you have a preferred \
day or week in mind, or would you like the earliest available?" \
Skip this if the patient already mentioned a time preference in the conversation.
Step 2: Call find_slots with the specialty and their preference. If they said \
"no preference" or "as soon as possible", call find_slots without a day preference.
Step 3: Look at the results and extract the UNIQUE DAYS that have availability. \
Present only the available days: "We have openings on Monday April 20th, \
Tuesday April 21st, and Wednesday April 22nd. Which day works best?" \
If only one day is available, skip to Step 4 with that day.
Step 4: Once the patient picks a day, ask: "Would you prefer morning or afternoon?"
Step 5: From the slots you already have, pick 2-3 options on that day matching \
their time preference. Present concisely. If it's the same doctor, mention the \
name once: "Dr. Rodriguez is available at 9:00 AM, 10:15 AM, or 11:00 AM."
If no slots match the requested time preference on that day, say that FIRST, then \
ask whether they'd like a different day or the other time of day. Do NOT list \
times from the other bucket unless the patient agrees to switch or asks to hear them.
Step 6: Once the patient picks a time, confirm ALL details before booking — doctor \
name, specialty, full date (day of week + month + day number), and time. \
Example: "I'll book you with Dr. Rodriguez for neurology on Wednesday, April 22nd \
at 10:15 AM. Shall I confirm?"
Step 7: Only call book_appointment AFTER the patient explicitly confirms.

## When no slots are found
- If a specific date has no slots, try ONE broader search (drop the day preference).
- If the broader search also returns nothing, be honest: "I'm sorry, there are no \
available [specialty] appointments at the moment." Then offer concrete options: \
trying a different specialty, or checking back another time.
- Do NOT ask the patient to try different days or be more flexible if you already \
searched without any date filter — that means nothing is available at all.
- Two searches maximum for the same specialty — if both return empty, move on.

## For reschedules
1. First ask: "Do you remember which doctor or specialty the appointment is with?" \
This helps narrow the search. Skip if the patient already mentioned it.
2. Call find_appointment to find their existing appointments. You can filter by \
doctor_name or specialty_name if the patient gives one. Always include the \
current patient's ID.
3. If ONE appointment is found: confirm it — "I see you have an appointment with \
Dr. Rodriguez on Thursday, April 23rd. Is that the one you'd like to reschedule?"
4. If MULTIPLE appointments are found: list them and ask which one — "I see you \
have 2 upcoming appointments: 1) Dr. Rodriguez on Thursday April 23rd, \
2) Dr. Watson on Monday April 27th. Which one would you like to reschedule?"
5. If NO appointments are found: let the patient know and offer to book a new one.
6. If the patient asks whether they have any other appointments, use find_appointment \
and answer that directly before continuing.
7. Once the patient confirms which appointment to move, call reschedule_appointment \
with the same patient_id to PREVIEW replacement options. Pass along any day or week \
preference the patient already gave. This preview does NOT cancel the current \
appointment.
8. If the preview spans multiple days, present only the available days and ask \
which day works best.
9. Once the patient picks a day, ask: "Would you prefer morning or afternoon?"
10. If the patient narrows or changes the requested day or time after a preview, \
ALWAYS call reschedule_appointment AGAIN with the SAME appointment_id, the SAME \
patient_id, plus the updated preferred_day and/or preferred_time before answering. \
Never infer filtered availability from an older preview.
11. If the requested time bucket has no matches on the chosen day, say that first \
("I don't see afternoon availability on Monday, April 20th."), then ask whether \
they'd like a different day or the other time of day. Do NOT list morning times \
until the patient agrees to switch or asks to hear them.
12. If same-doctor options do not work, use find_slots with the specialty_id from the \
preview to search other doctors in that specialty.
13. Only after the patient explicitly confirms the replacement slot, call \
reschedule_appointment again with appointment_id, patient_id, new_doctor_id, \
new_specialty_id, new_start_at, and new_end_at to finalize the reschedule.

## For cancellations
1. First ask: "Do you remember which doctor or specialty the appointment is with?" \
Skip if already mentioned.
2. Call find_appointment to find their existing appointments, always using the \
current patient's ID.
3. If ONE: confirm — "I see your appointment with Dr. Rodriguez on Thursday, \
April 23rd. Would you like to cancel this one?"
4. If MULTIPLE: list them and ask which one.
5. If NONE: let the patient know.
6. Only call cancel_appointment with patient_id and appointment_id AFTER the \
patient explicitly confirms.

## Rules
- ALWAYS ask for explicit confirmation before booking, rescheduling, or cancelling.
- ALWAYS include the full date (day + month + day number) when presenting days or times.
- NEVER present more than 3 options at a time.
- Ask ONE question per message. Never combine day + time in one question.
- NEVER present a slot that does not match the patient's stated day or time preference.
- If the requested morning/afternoon bucket has no matches, say that before offering alternatives.
- Do NOT list times from a different bucket until the patient asks for them or agrees to switch.
- Do NOT fabricate appointment times or doctor names — only use data from tool results.
- Always include the current patient's ID in find_appointment, reschedule_appointment, and cancel_appointment.
- Do NOT give medical advice.
- Keep responses to 1-2 sentences, except when listing options.
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


def _message_text(content: object) -> str:
    """Flatten message content blocks into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    return ""


def _parse_patient_identity(content: str) -> tuple[str, str] | None:
    """Extract ``(patient_id, patient_name)`` from intake tool output."""
    if "ID: " not in content or " (" not in content:
        return None

    for prefix in ("Patient found: ", "Successfully registered "):
        if prefix not in content:
            continue

        name_start = content.index(prefix) + len(prefix)
        name_end = content.index(" (", name_start)
        id_start = content.index("ID: ") + 4
        id_end = content.index(",", id_start)
        return (
            content[id_start:id_end].strip(),
            content[name_start:name_end].strip(),
        )

    return None


def _is_affirmative_identity_reply(content: object) -> bool:
    """Return True when the patient's latest reply confirms a matched record."""
    normalized = " ".join(_message_text(content).lower().split())
    if not normalized:
        return False

    tokens = normalized.split()

    negative_tokens = {"no", "nope", "nah"}
    if any(token in tokens for token in negative_tokens):
        return False

    negative_phrases = (
        "not me",
        "wrong person",
        "wrong patient",
        "wrong uin",
        "different uin",
    )
    if any(marker in normalized for marker in negative_phrases):
        return False

    affirmative_tokens = {"yes", "yeah", "yep", "yup", "correct"}
    if any(token in tokens for token in affirmative_tokens):
        return True

    affirmative_phrases = (
        "that's me",
        "that is me",
        "yes that's me",
        "yes that is me",
    )
    return any(marker in normalized for marker in affirmative_phrases)


def _confirmed_patient_from_history(messages: list[Any]) -> tuple[str, str] | None:
    """Recover the latest matched patient once the human explicitly confirms."""
    latest_human_index: int | None = None

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if getattr(message, "type", None) == "human":
            latest_human_index = index
            break

    if latest_human_index is None:
        return None

    latest_human = messages[latest_human_index]
    if not _is_affirmative_identity_reply(getattr(latest_human, "content", "")):
        return None

    for message in reversed(messages[:latest_human_index]):
        if getattr(message, "type", None) != "tool":
            continue
        content = _message_text(getattr(message, "content", ""))
        if "Patient found:" not in content:
            continue
        return _parse_patient_identity(content)

    return None


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

    After the agent finishes, we scan the ToolMessages to find the
    patient_id and name so we can set them in shared state.

    Extraction targets:
      - "Patient found: Sarah (ID: abc-123, UIN: 123456789)"
      - "Successfully registered Sarah (ID: abc-123, UIN: 123456789)"

    Existing-patient lookups are only committed AFTER the patient confirms
    "yes, that's me." New registrations are committed immediately.
    """
    agent = _get_intake_agent()
    result = await agent.ainvoke({"messages": state["messages"]})
    all_messages = result["messages"]
    input_count = len(state["messages"])
    fresh_messages = all_messages[input_count:]

    patient_id = state.get("patient_id")
    patient_name = state.get("patient_name")

    if patient_id is None:
        confirmed_patient = _confirmed_patient_from_history(state["messages"])
        if confirmed_patient is not None:
            patient_id, patient_name = confirmed_patient

    for msg in fresh_messages:
        if not hasattr(msg, "type") or msg.type != "tool":
            continue

        content = _message_text(msg.content)
        if "Successfully registered" not in content:
            continue

        parsed_identity = _parse_patient_identity(content)
        if parsed_identity is None:
            continue
        patient_id, patient_name = parsed_identity

    return {
        "messages": fresh_messages,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "last_agent": "intake",
    }


async def triage_node(state: AgentState) -> dict:
    """Run the Triage Agent and extract specialty match from results.

    After the agent collects symptoms and runs triage_symptoms,
    we extract:
      - specialty_id: from the tool results (top-ranked match)
      - symptoms: from the tool call arguments (the structured list
        the LLM extracted from the patient's description)

    The triage tool returns lines like:
      "- Cardiology (ID: abc-123): score 8.50, matched on: chest pain"
    We take the first match as the primary recommendation.
    """
    agent = _get_triage_agent()
    result = await agent.ainvoke({"messages": state["messages"]})
    all_messages = result["messages"]
    input_count = len(state["messages"])
    fresh_messages = all_messages[input_count:]

    specialty_id = state.get("specialty_id")
    symptoms = list(state.get("symptoms", []))

    for msg in fresh_messages:
        if not hasattr(msg, "type"):
            continue

        # ── Extract specialty from triage tool results ───
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

    return {
        "messages": fresh_messages,
        "specialty_id": specialty_id,
        "symptoms": symptoms,
        "last_agent": "triage",
    }


async def scheduling_node(state: AgentState) -> dict:
    """Run the Scheduling Agent and extract appointment state from results.

    The Scheduling Agent handles booking, rescheduling, and cancelling.
    After it runs, we extract:
      - appointment_id: the booked or rescheduled appointment after a
        successful final action
      - selected_appointment_id: the existing appointment the patient
        chose for reschedule/cancel, pulled from structured tool-call args
    """
    agent = _get_scheduling_agent()
    result = await agent.ainvoke({"messages": state["messages"]})
    all_messages = result["messages"]
    input_count = len(state["messages"])
    fresh_messages = all_messages[input_count:]

    appointment_id = state.get("appointment_id")
    selected_appointment_id = state.get("selected_appointment_id")
    if state.get("intent") == "book":
        selected_appointment_id = None

    # Track whether the FINAL action completed this turn.
    # For bookings: book_appointment succeeded
    # For cancellations: cancel_appointment succeeded
    action_completed = False

    for msg in fresh_messages:
        if not hasattr(msg, "type"):
            continue

        if msg.type == "ai" and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc["name"] not in ("reschedule_appointment", "cancel_appointment"):
                    continue
                args = tc.get("args", {})
                appointment_arg = args.get("appointment_id")
                if isinstance(appointment_arg, str) and appointment_arg.strip():
                    selected_appointment_id = appointment_arg.strip()

        if msg.type != "tool":
            continue

        # Detect successful booking or finalized reschedule
        if (
            "Appointment ID: " in msg.content
            and (
                "booked successfully" in msg.content
                or "rescheduled successfully" in msg.content
            )
        ):
            id_start = msg.content.index("Appointment ID: ") + 16
            id_end = msg.content.index(".", id_start)
            appointment_id = msg.content[id_start:id_end].strip()
            action_completed = True

        # Detect successful standalone cancellation (cancel_appointment tool)
        if "has been cancelled" in msg.content:
            action_completed = True

    result = {
        "messages": fresh_messages,
        "appointment_id": appointment_id,
        "selected_appointment_id": selected_appointment_id,
        "last_agent": "scheduling",
    }

    # Only reset intent after the conversation's final action.
    # This prevents premature reset during multi-step flows
    # like reschedule (which cancels then books).
    if action_completed:
        result["intent"] = None

    return result
