from typing import Annotated, Literal
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state that flows through all agents in the graph.

    Fields fall into two categories:
      - Conversation: the message history (append-only)
      - Handoff data: structured fields that agents set and the
        supervisor reads to make routing decisions
    """

    # ── Conversation history ─────────────────────────────────
    # Annotated with add_messages so LangGraph appends new
    # messages instead of replacing the list.
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Patient identity (set by Intake Agent) ───────────────
    patient_id: str | None
    patient_name: str | None

    # ── Triage results (set by Triage Agent) ─────────────────
    symptoms: list[str]
    specialty_id: str | None

    # ── Scheduling context (set by Scheduling Agent) ─────────
    appointment_id: str | None

    # ── Routing control (set by Supervisor) ──────────────────
    current_agent: str
    intent: Literal["book", "reschedule", "cancel"] | None