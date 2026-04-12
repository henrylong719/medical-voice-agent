"""
Shared state definition for the multi-agent medical scheduling system.

AgentState is a TypedDict that flows through all nodes in the graph.
Every sub-agent reads from it and writes back to it. The Supervisor
reads state fields to make routing decisions.

Fields fall into two categories:
  - Conversation: the message history (append-only via add_messages)
  - Handoff data: structured fields that accumulate as agents do work

The add_messages annotation on the messages field tells LangGraph to
APPEND new messages rather than replacing the list. All other fields
use the default behavior: new values overwrite old ones.
"""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state that flows through all agents in the graph.

    Attributes:
        messages: Full conversation history. Annotated with add_messages
            so LangGraph appends new messages instead of replacing.
        patient_id: UUID of the identified patient. Set by Intake Agent
            after a successful patient lookup confirmation or register_patient call.
        patient_name: Patient's full name for personalized responses.
            Set by Intake Agent alongside patient_id.
        patient_status: Whether the patient says they are new to the
            clinic or returning. Set by the Supervisor during booking
            flows so Intake can choose the right verification path.
        symptoms: List of symptoms collected during triage. Set by
            Triage Agent from the triage_symptoms tool call arguments.
        specialty_id: UUID of the matched specialty. Set by Triage Agent
            after triage_symptoms returns results. The Supervisor checks
            this to know when triage is complete.
        appointment_id: UUID of the appointment affected by the most
            recent successful book/reschedule action. Set by Scheduling
            Agent after a successful book_appointment or finalized
            reschedule_appointment call.
        selected_appointment_id: UUID of the existing appointment being
            worked on in a reschedule/cancel flow. Set by Scheduling Agent
            from structured tool-call arguments once a specific appointment
            has been chosen.
        current_agent: Which sub-agent is active or should run next.
            Set by Supervisor to control routing. Values: "intake",
            "triage", "scheduling", "supervisor" (re-run), or "done"
            (wait for patient input).
        intent: What the patient wants to do. Classified by Supervisor
            from the conversation when first needed. Used to decide
            whether triage is needed (book) or not (reschedule/cancel).
    """

    # ── Conversation history ─────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Patient identity (set by Intake Agent) ───────────────
    patient_id: str | None
    patient_name: str | None
    patient_status: Literal["new", "returning"] | None

    # ── Triage results (set by Triage Agent) ─────────────────
    symptoms: list[str]
    specialty_id: str | None

    # ── Scheduling context (set by Scheduling Agent) ─────────
    appointment_id: str | None
    selected_appointment_id: str | None

    # ── Routing control (set by Supervisor) ──────────────────
    current_agent: str
    intent: Literal["book", "reschedule", "cancel"] | None

    # ── Loop prevention (set by sub-agent nodes) ─────────────
    # Tracks which sub-agent just ran. The Supervisor checks this
    # to avoid re-routing to an agent that is already waiting for
    # patient input, which would cause an infinite loop.
    last_agent: str | None
