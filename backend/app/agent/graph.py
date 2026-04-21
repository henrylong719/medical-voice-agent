"""
Multi-agent medical scheduling graph built with LangGraph.

Replaces the Phase 2 monolithic agent with a Supervisor + 3 sub-agents:
  - Supervisor: routes to the right sub-agent based on state
  - Intake Agent: identifies or registers patients
  - Triage Agent: collects symptoms, matches to specialty via RAG
  - Scheduling Agent: books, reschedules, cancels appointments

The graph loop:
  Patient message → Supervisor → Sub-agent → Supervisor → ...
  When waiting for patient input, the graph exits to END.

Graph structure:
    START → supervisor ←→ intake
                       ←→ triage
                       ←→ scheduling
                       → END (when waiting for patient)

Persistence is handled by AsyncPostgresSaver on the outer graph,
so conversation state survives across messages and server restarts.
The sub-agents do NOT have their own checkpointers — they read from
and write to the shared AgentState.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator
from urllib.parse import quote, urlparse, urlunparse

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from app.agent.agents import intake_node, triage_node, scheduling_node
from app.agent.guardrails import check_output, sanitize_output
from app.agent.state import AgentState
from app.agent.supervisor import supervisor_node
from app.config import settings

logger = logging.getLogger(__name__)


class AgentConfigurationError(RuntimeError):
    """Raised when required backend settings for the agent are missing."""


# ============================================================
# CHECKPOINTER (persistent conversation memory)
# ============================================================
# Same pattern as Phase 2 — AsyncPostgresSaver stores conversation
# state in Supabase Postgres. Now it wraps the OUTER graph, not a
# single agent. Sub-agents don't need their own checkpointers
# because they read/write through the shared AgentState.
# ============================================================

_checkpointer: AsyncPostgresSaver | None = None
_checkpointer_stack: AsyncExitStack | None = None
_checkpointer_loop: asyncio.AbstractEventLoop | None = None


def _safe_db_uri(uri: str) -> str:
    """URL-encode the password in a Postgres URI so special chars are handled."""
    parsed = urlparse(uri)
    if parsed.password:
        encoded_password = quote(parsed.password, safe="")
        netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunparse(parsed._replace(netloc=netloc))
    return uri


async def _get_checkpointer() -> AsyncPostgresSaver:
    """Get or create the AsyncPostgresSaver singleton."""
    global _checkpointer, _checkpointer_stack, _checkpointer_loop

    running_loop = asyncio.get_running_loop()

    db_uri = settings.supabase_db_uri.strip()
    if not db_uri:
        raise AgentConfigurationError(
            "SUPABASE_DB_URI is not configured. Set it in backend/.env before "
            "using the chat API."
        )

    if _checkpointer is not None and _checkpointer_loop is not running_loop:
        logger.info("Event loop changed; resetting cached checkpointer")
        await cleanup_checkpointer()

    if _checkpointer is None:
        stack = AsyncExitStack()
        try:
            conn = await stack.enter_async_context(
                await AsyncConnection[dict[str, Any]].connect(
                    _safe_db_uri(db_uri),
                    autocommit=True,
                    prepare_threshold=None,
                    row_factory=dict_row,
                )
            )
            checkpointer = AsyncPostgresSaver(conn=conn)
            await checkpointer.setup()
        except Exception:
            await stack.aclose()
            raise

        _checkpointer = checkpointer
        _checkpointer_stack = stack
        _checkpointer_loop = running_loop
        logger.info("PostgresSaver checkpointer initialized")

    return _checkpointer


async def cleanup_checkpointer() -> None:
    """Close the checkpointer connection and clear cached state."""
    global _graph, _graph_loop, _checkpointer, _checkpointer_loop, _checkpointer_stack

    if _checkpointer_stack is not None:
        try:
            await _checkpointer_stack.aclose()
        except Exception:
            logger.warning(
                "Failed to close cached PostgresSaver cleanly; resetting state anyway",
                exc_info=True,
            )
        logger.info("PostgresSaver checkpointer closed")

    _checkpointer_stack = None
    _checkpointer = None
    _checkpointer_loop = None
    _graph = None
    _graph_loop = None


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================


def _route_from_supervisor(state: AgentState) -> str:
    """Conditional edge: read current_agent to decide the next node.

    Called after the Supervisor runs. The Supervisor sets current_agent
    to tell the graph where to go:
      - "intake" / "triage" / "scheduling" → route to that sub-agent
      - "supervisor" → Supervisor wants to re-run itself (e.g., after
        classifying intent, it needs to re-evaluate routing rules)
      - "done" or anything else → the Supervisor responded directly
        to the patient (e.g., asked a clarifying question), so we go
        to END and wait for the next message
    """
    next_agent = state.get("current_agent", "")

    if next_agent in ("intake", "triage", "scheduling", "supervisor"):
        return next_agent

    # Supervisor added a message for the patient — wait for reply
    return END


def _build_graph() -> StateGraph:
    """Construct the multi-agent state graph.

    Nodes:
      - supervisor: inspects state, routes to sub-agents or END
      - intake: identifies / registers patients
      - triage: collects symptoms, matches to specialty
      - scheduling: books / reschedules / cancels appointments

    Edges:
      - START → supervisor (entry point for every message)
      - supervisor → intake / triage / scheduling (conditional)
      - supervisor → supervisor (re-run after intent classification)
      - supervisor → END (waiting for patient input)
      - intake / triage / scheduling → supervisor (always return)
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("intake", intake_node)
    graph.add_node("triage", triage_node)
    graph.add_node("scheduling", scheduling_node)

    # ── Entry point ──────────────────────────────────────
    # Every patient message enters through the Supervisor
    graph.add_edge(START, "supervisor")

    # ── Supervisor routing (conditional edges) ───────────
    # After the Supervisor runs, _route_from_supervisor
    # reads current_agent to decide where to go next.
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "intake": "intake",
            "triage": "triage",
            "scheduling": "scheduling",
            "supervisor": "supervisor",
            END: END,
        },
    )

    # ── Sub-agent → Supervisor (fixed edges) ─────────────
    # After any sub-agent finishes, always return to
    # Supervisor. It re-evaluates state and decides what's
    # next — route to another agent, or go to END.
    graph.add_edge("intake", "supervisor")
    graph.add_edge("triage", "supervisor")
    graph.add_edge("scheduling", "supervisor")

    return graph


# ============================================================
# COMPILED GRAPH (with checkpointer)
# ============================================================

_graph: Any | None = None
_graph_loop: asyncio.AbstractEventLoop | None = None


async def _get_or_build_graph() -> Any:
    """Lazy singleton: build and compile the graph on first request."""
    global _graph, _graph_loop

    running_loop = asyncio.get_running_loop()
    if _graph is not None and _graph_loop is not running_loop:
        logger.info("Event loop changed; resetting cached compiled graph")
        await cleanup_checkpointer()

    if _graph is None:
        api_key = settings.anthropic_api_key.strip()
        if not api_key:
            raise AgentConfigurationError(
                "ANTHROPIC_API_KEY is not configured. Set it in backend/.env "
                "before using the chat API."
            )

        checkpointer = await _get_checkpointer()
        graph = _build_graph()
        _graph = graph.compile(checkpointer=checkpointer)
        _graph_loop = running_loop
        logger.info("Multi-agent graph compiled")

    return _graph


async def ensure_agent_ready() -> None:
    """Eagerly validate configuration and initialize the graph singleton."""
    await _get_or_build_graph()


# ============================================================
# TEXT EXTRACTION HELPER
# ============================================================


def _extract_text_content(content: Any) -> str:
    """Normalize LangChain message content into plain text.

    LangChain message .content can be either:
      - A plain string: "Welcome back, Sarah!"
      - A list of content blocks: [{"type": "text", "text": "..."}, ...]

    Extracts only text portions, skipping tool call blocks that
    the patient shouldn't see.
    """
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


# ============================================================
# PUBLIC API
# ============================================================
# These functions are the interface that chat/routes.py calls.
# The signatures are identical to Phase 2 — the multi-agent
# complexity is entirely hidden behind the same API.
# ============================================================


async def stream_agent_response(
    message: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """Send a message to the multi-agent graph and stream the response.

    Args:
        message: The patient's message text.
        thread_id: Unique conversation ID. Same thread_id = same
            conversation history (persisted via checkpointer).

    Yields:
        Text chunks as they're generated by whichever agent responds.

    After streaming completes, runs output guardrails on the full
    response text. Violations are logged as warnings for prompt
    improvement — we can't un-stream tokens already sent.
    In Phase 6 (voice), the TTS buffering step will allow real-time
    interception before audio is synthesized.
    """
    graph = await _get_or_build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("human", message)]}

    # Collect the full response for post-stream scanning
    streamed_chunks: list[str] = []

    async for event in graph.astream_events(
        input_message,
        config=config,
        version="v2",
    ):
        if event["event"] != "on_chat_model_stream":
            continue

        data = event.get("data")
        if not isinstance(data, dict):
            continue

        chunk = data.get("chunk")
        text = _extract_text_content(getattr(chunk, "content", None))
        if text:
            streamed_chunks.append(text)
            yield text

    # ── Post-stream output guardrail ──────────────────────
    # Scan the full response for medical advice violations.
    # We can't rewrite what was already streamed, but we log
    # violations so they surface in monitoring and drive
    # prompt improvements.
    full_response = "".join(streamed_chunks)
    if full_response:
        violations = check_output(full_response)
        if violations:
            logger.warning(
                "OUTPUT GUARDRAIL (post-stream): %d violation(s) in "
                "streamed response for thread %s: %s",
                len(violations),
                thread_id,
                ", ".join(v.matched_text for v in violations),
            )


async def invoke_agent(message: str, thread_id: str) -> str:
    """Send a message and get the complete response (non-streaming).

    Useful for testing and for contexts where streaming isn't needed.
    """
    graph = await _get_or_build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("human", message)]}

    result = await graph.ainvoke(input_message, config=config)

    # Count how many messages were in state before this invocation.
    # The input adds 1 human message, so new agent messages start
    # after that. We look at the full result and collect only the
    # patient-facing AI messages added during THIS turn.
    #
    # We find the input human message by ID and collect everything after it.
    messages = result.get("messages", [])

    # Find where the new human message is — everything after it is new
    new_start = 0
    for i, msg in enumerate(messages):
        if hasattr(msg, "type") and msg.type == "human":
            new_start = i  # Keep updating — we want the LAST human message

    responses: list[str] = []
    for msg in messages[new_start + 1 :]:
        if not isinstance(msg, AIMessage):
            continue
        text = _extract_text_content(msg.content)
        if text.strip():
            responses.append(text.strip())

    full_response = "\n\n".join(responses)

    # ── Output guardrail: scan for medical advice ─────────
    # We have the complete response, so we can rewrite it
    # before the patient sees it.
    return sanitize_output(full_response)
