"""
Single voice-optimized agent for medical scheduling.

Replaces the Phase 4 multi-agent graph (supervisor + 3 sub-agents)
with a single agent that has all 10 tools and a consolidated
voice-optimized system prompt. This simplification lets us focus
on voice pipeline development without debugging orchestration
edge cases.

Architecture:
    Patient message → screen_input() → Agent (all tools) → sanitize_output() → Response

The public API is identical to the multi-agent version:
    - stream_agent_response(message, thread_id) → AsyncIterator[str]
    - invoke_agent(message, thread_id) → str
    - ensure_agent_ready() → None
    - cleanup_checkpointer() → None

chat/routes.py and the future voice pipeline call these functions
without knowing which agent architecture is behind the wall.

Swap-back plan:
    To restore the multi-agent system, revert this file and move
    screen_input() back into the supervisor node. Everything else
    (tools, services, guardrails) is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator
from urllib.parse import quote, urlparse, urlunparse

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic import SecretStr

from app.agent.guardrails import check_output, sanitize_output, screen_input
from app.agent.tools import (
    book_appointment,
    cancel_appointment,
    find_appointment,
    find_patient_by_identifier,
    find_patients_by_demographics,
    find_slots,
    list_specialties,
    register_patient,
    reschedule_appointment,
    triage_symptoms,
)
from app.agent.voice_prompt import VOICE_SYSTEM_PROMPT
from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentConfigurationError(RuntimeError):
    """Raised when required backend settings for the agent are missing."""


# ============================================================
# ALL TOOLS — same 10 tools the sub-agents used, now on one agent
# ============================================================

ALL_TOOLS = [
    find_patient_by_identifier,
    find_patients_by_demographics,
    register_patient,
    triage_symptoms,
    find_slots,
    book_appointment,
    find_appointment,
    reschedule_appointment,
    cancel_appointment,
    list_specialties,
]


# ============================================================
# CHECKPOINTER (persistent conversation memory)
# ============================================================
# Identical to the multi-agent version. AsyncPostgresSaver stores
# conversation state in Supabase Postgres. The single agent uses
# the same checkpointer — conversation history survives across
# messages and server restarts.
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

    db_uri = settings.SUPABASE_DB_URI.strip()
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
# AGENT CONSTRUCTION
# ============================================================
# Instead of a StateGraph with 4 nodes and conditional edges,
# we use create_agent directly. It builds its own internal graph
# with the observe → think → act loop. We just give it the LLM,
# tools, and system prompt.
# ============================================================

_graph: Any | None = None
_graph_loop: asyncio.AbstractEventLoop | None = None


def _build_llm() -> ChatAnthropic:
    """Build the Claude LLM instance."""
    return ChatAnthropic(
        model_name=settings.ANTHROPIC_MODEL,
        api_key=SecretStr(settings.ANTHROPIC_API_KEY),
        temperature=0,
        max_tokens_to_sample=1024,
        timeout=None,
        stop=None,
    )


async def _get_or_build_graph() -> Any:
    """Lazy singleton: build and compile the agent on first request."""
    global _graph, _graph_loop

    running_loop = asyncio.get_running_loop()
    if _graph is not None and _graph_loop is not running_loop:
        logger.info("Event loop changed; resetting cached compiled graph")
        await cleanup_checkpointer()

    if _graph is None:
        api_key = settings.ANTHROPIC_API_KEY.strip()
        if not api_key:
            raise AgentConfigurationError(
                "ANTHROPIC_API_KEY is not configured. Set it in backend/.env "
                "before using the chat API."
            )

        checkpointer = await _get_checkpointer()

        # Build the single agent with all tools and the voice prompt.
        # create_agent returns a CompiledStateGraph — it already
        # includes the observe → think → act loop internally.
        # We pass the checkpointer directly so conversation history
        # persists across messages and server restarts.
        _graph = create_agent(
            model=_build_llm(),
            tools=ALL_TOOLS,
            system_prompt=VOICE_SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        _graph_loop = running_loop
        logger.info("Single voice agent compiled")

    return _graph


async def ensure_agent_ready() -> None:
    """Eagerly validate configuration and initialize the agent singleton."""
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
# Same signatures as the multi-agent version. chat/routes.py
# and the future voice pipeline call these without changes.
#
# Key difference from multi-agent: screen_input() now runs HERE
# instead of inside the supervisor node. This is the boundary
# layer — every message passes through these functions, so
# guardrails are guaranteed to run on every turn.
# ============================================================


async def stream_agent_response(
    message: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """Send a message to the agent and stream the response.

    Args:
        message: The patient's message text.
        thread_id: Unique conversation ID. Same thread_id = same
            conversation history (persisted via checkpointer).

    Yields:
        Text chunks as they're generated by the agent.

    Guardrails:
        - Input: screen_input() runs before the agent. If triggered,
          yields the guardrail response and returns immediately.
        - Output: after streaming completes, scans the full response
          for medical advice violations. Can't un-stream tokens, but
          logs violations for monitoring and prompt improvement.
    """
    # ── Input guardrail: intercept before the agent sees it ──
    guardrail_result = screen_input(message)
    if guardrail_result is not None:
        logger.warning(
            "Input guardrail triggered (%s) for thread %s — short-circuiting",
            guardrail_result.category,
            thread_id,
        )
        yield guardrail_result.response
        return

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

    Guardrails:
        - Input: screen_input() runs before the agent. If triggered,
          returns the guardrail response immediately.
        - Output: scans the full response and rewrites it if medical
          advice violations are found.
    """
    # ── Input guardrail: intercept before the agent sees it ──
    guardrail_result = screen_input(message)
    if guardrail_result is not None:
        logger.warning(
            "Input guardrail triggered (%s) for thread %s — short-circuiting",
            guardrail_result.category,
            thread_id,
        )
        return guardrail_result.response

    graph = await _get_or_build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("human", message)]}

    result = await graph.ainvoke(input_message, config=config)

    # Find the last human message and collect all AI responses after it.
    messages = result.get("messages", [])

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

    # ── Output guardrail: scan and rewrite if needed ──────
    return sanitize_output(full_response)
