"""
Medical scheduling agent built with LangGraph.

This module creates the agent "graph" — the loop where the LLM:
  1. Reads the conversation + tool descriptions
  2. Decides to call a tool or respond to the patient
  3. If it called a tool, reads the result and loops back to step 1
  4. If it responded, the loop ends (until the next patient message)

Key components:
  - ChatAnthropic: the Claude LLM with tools bound
  - AsyncPostgresSaver: persists conversation history in Supabase Postgres
  - create_agent: LangChain's agent factory (built on LangGraph)
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, cast
from urllib.parse import quote, urlparse, urlunparse

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tools import ALL_TOOLS
from app.config import settings

logger = logging.getLogger(__name__)


class AgentConfigurationError(RuntimeError):
    """Raised when required backend settings for the agent are missing."""


# ============================================================
# CHECKPOINTER (persistent conversation memory)
# ============================================================
# AsyncPostgresSaver stores conversation state in your Supabase
# Postgres database. Each conversation thread gets its own
# checkpoint, so:
#   - Conversations survive server restarts
#   - Multiple server instances share the same memory
#   - You can inspect conversation state directly in the DB
#
# setup() creates the checkpoint tables automatically the first
# time we initialize the saver.
# ============================================================

_checkpointer: AsyncPostgresSaver | None = None
_checkpointer_stack: AsyncExitStack | None = None


def _safe_db_uri(uri: str) -> str:
    """URL-encode the password in a Postgres URI so special chars like % are handled."""
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
    global _checkpointer, _checkpointer_stack

    db_uri = settings.supabase_db_uri.strip()
    if not db_uri:
        raise AgentConfigurationError(
            "SUPABASE_DB_URI is not configured. Set it in backend/.env before "
            "using the chat API."
        )

    if _checkpointer is None:
        stack = AsyncExitStack()
        try:
            conn = await stack.enter_async_context(
                await AsyncConnection.connect(  
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
        logger.info("PostgresSaver checkpointer initialized")

    return _checkpointer


async def cleanup_checkpointer() -> None:
    """Close the checkpointer connection and clear cached state."""
    global _agent, _checkpointer, _checkpointer_stack

    if _checkpointer_stack is not None:
        await _checkpointer_stack.aclose()
        _checkpointer_stack = None
        _checkpointer = None
        _agent = None
        logger.info("PostgresSaver checkpointer closed")


# ============================================================
# AGENT
# ============================================================

def _build_llm() -> ChatAnthropic:
    """
    Build the Claude LLM instance.

    We use Claude Haiku 4.5 because it's fast and cheap — ideal for
    a scheduling agent that needs quick responses but doesn't require
    deep reasoning.
    """
    api_key = settings.anthropic_api_key.strip()
    if not api_key:
        raise AgentConfigurationError(
            "ANTHROPIC_API_KEY is not configured. Set it in backend/.env before "
            "using the chat API."
        )

    return ChatAnthropic(
        model_name=settings.anthropic_model,
        api_key=SecretStr(api_key),
        temperature=0,
        max_tokens_to_sample=1024,
        timeout=None,
        stop=None,
    )


async def _get_agent() -> Any:
    """
    Build the agent with persistent checkpointer.

    create_agent sets up the full observe-think-act loop:
      - Binds tools to the LLM so it knows what's available
      - Handles the tool call -> execute -> feed result back cycle
      - Uses AsyncPostgresSaver to persist messages across turns
    """
    checkpointer = await _get_checkpointer()

    return create_agent(
        model=_build_llm(),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


_agent: Any | None = None


async def _get_or_build_agent() -> Any:
    """Lazy singleton: build the agent on first request."""
    global _agent
    if _agent is None:
        _agent = await _get_agent()
    return _agent


async def ensure_agent_ready() -> None:
    """Eagerly validate configuration and initialize the agent singleton."""
    await _get_or_build_agent()


def _extract_text_content(content: Any) -> str:
    """
    Normalize LangChain message content into plain text.

    LangChain message .content can be either:
      - A plain string: "Welcome back, Sarah!"
      - A list of content blocks: [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]

    This helper handles both cases and extracts only the text portions,
    skipping tool call blocks that the patient shouldn't see.
    """
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


# ============================================================
# PUBLIC API
# ============================================================

async def stream_agent_response(
    message: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """
    Send a message to the agent and stream the response token by token.

    Args:
        message: The patient's message text.
        thread_id: Unique conversation ID. Same thread_id means the same
            conversation history.

    Yields:
        Text chunks as they're generated by the LLM.
    """
    agent = await _get_or_build_agent()
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("human", message)]}

    async for event in agent.astream_events(
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
            yield text


async def invoke_agent(message: str, thread_id: str) -> str:
    """
    Send a message and get the complete response (non-streaming).

    Useful for testing and for contexts where streaming isn't needed.
    """
    agent = await _get_or_build_agent()
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [("human", message)]}

    result = cast(dict[str, Any], await agent.ainvoke(input_message, config=config))
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""

    last_message = messages[-1]
    return _extract_text_content(getattr(last_message, "content", None))
