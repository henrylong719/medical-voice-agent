from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pytest import MonkeyPatch

from app.agent import graph as agent_graph
from app.agent import supervisor
from app.agent.state import AgentState

WorkflowNode = Callable[[AgentState], Awaitable[dict[str, Any]]]
IntentClassifier = Callable[[AgentState], Awaitable[str | None]]


def flatten_message_content(content: object) -> str:
    """Convert message content into plain text for test assertions."""
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
    return str(content)


def latest_human_text(state: AgentState) -> str:
    """Return the newest human utterance in the workflow state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return flatten_message_content(message.content)
    return ""


def ai_history_contains(state: AgentState, snippet: str) -> bool:
    """Check whether any AI message in the conversation contains text."""
    normalized_snippet = snippet.lower()
    for message in state["messages"]:
        if not isinstance(message, AIMessage):
            continue
        if normalized_snippet in flatten_message_content(message.content).lower():
            return True
    return False


async def default_intent_classifier(state: AgentState) -> str | None:
    """Deterministic intent classifier for workflow tests."""
    latest = latest_human_text(state).lower()
    if "reschedule" in latest or "move" in latest:
        return "reschedule"
    if "cancel" in latest:
        return "cancel"
    if (
        "book" in latest
        or "appointment" in latest
        or "headache" in latest
        or "dizziness" in latest
    ):
        return "book"
    return None


def install_test_graph(
    monkeypatch: MonkeyPatch,
    *,
    intake_node: WorkflowNode,
    triage_node: WorkflowNode,
    scheduling_node: WorkflowNode,
    classify_intent: IntentClassifier | None = None,
) -> None:
    """Compile the real graph with fake sub-agents and in-memory state."""
    monkeypatch.setattr(
        supervisor,
        "_classify_intent",
        classify_intent or default_intent_classifier,
    )
    monkeypatch.setattr(agent_graph, "intake_node", intake_node)
    monkeypatch.setattr(agent_graph, "triage_node", triage_node)
    monkeypatch.setattr(agent_graph, "scheduling_node", scheduling_node)

    compiled_graph = agent_graph._build_graph().compile(checkpointer=InMemorySaver())

    async def fake_get_or_build_graph():
        return compiled_graph

    monkeypatch.setattr(agent_graph, "_get_or_build_graph", fake_get_or_build_graph)


def invoke_turn(message: str, thread_id: str) -> str:
    """Send one message through the workflow test graph."""
    return asyncio.run(agent_graph.invoke_agent(message, thread_id))


def invoke_sequence(thread_id: str, *messages: str) -> list[str]:
    """Run a whole conversation and collect each agent response."""
    return [invoke_turn(message, thread_id) for message in messages]
