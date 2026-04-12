from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from pytest import MonkeyPatch

from app.agent import graph
from app.agent.state import AgentState
from tests.support import FakeCompiledGraph


def test_safe_db_uri_encodes_password() -> None:
    result = graph._safe_db_uri(
        "postgresql://user:pa:ss word@localhost:5432/postgres"
    )

    assert result == (
        "postgresql://user:pa%3Ass%20word@localhost:5432/postgres"
    )


def test_route_from_supervisor_returns_expected_edge() -> None:
    triage_state = cast(AgentState, {"current_agent": "triage"})
    done_state = cast(AgentState, {"current_agent": "done"})
    assert graph._route_from_supervisor(triage_state) == "triage"
    assert graph._route_from_supervisor(done_state) == END


def test_graph_extract_text_content_handles_supported_shapes() -> None:
    assert graph._extract_text_content("hello") == "hello"
    assert graph._extract_text_content(
        ["hi ", {"text": "there"}, {"type": "tool_use"}]
    ) == "hi there"
    assert graph._extract_text_content(None) == ""


def test_get_or_build_graph_requires_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "_graph", None)
    monkeypatch.setattr(graph.settings, "anthropic_api_key", "", raising=False)

    with pytest.raises(graph.AgentConfigurationError):
        asyncio.run(graph._get_or_build_graph())


def test_get_or_build_graph_compiles_once(monkeypatch: MonkeyPatch) -> None:
    class FakeBuilder:
        def __init__(self) -> None:
            self.compile_calls: list[object] = []

        def compile(self, *, checkpointer: object) -> str:
            self.compile_calls.append(checkpointer)
            return "compiled-graph"

    builder = FakeBuilder()

    async def fake_get_checkpointer() -> str:
        return "checkpointer"

    monkeypatch.setattr(graph, "_graph", None)
    monkeypatch.setattr(graph.settings, "anthropic_api_key", "test-key", raising=False)
    monkeypatch.setattr(graph, "_get_checkpointer", fake_get_checkpointer)
    monkeypatch.setattr(graph, "_build_graph", lambda: builder)

    async def run_twice() -> tuple[str, str]:
        first = await graph._get_or_build_graph()
        second = await graph._get_or_build_graph()
        return first, second

    first, second = asyncio.run(run_twice())

    assert first == "compiled-graph"
    assert second == "compiled-graph"
    assert builder.compile_calls == ["checkpointer"]


def test_get_or_build_graph_rebuilds_after_event_loop_change(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeBuilder:
        def __init__(self) -> None:
            self.compile_calls: list[object] = []

        def compile(self, *, checkpointer: object) -> str:
            self.compile_calls.append(checkpointer)
            return f"compiled-{len(self.compile_calls)}"

    builder = FakeBuilder()
    checkpointers = iter(["checkpointer-1", "checkpointer-2"])

    async def fake_get_checkpointer() -> str:
        return next(checkpointers)

    async def fake_cleanup() -> None:
        graph._graph = None
        graph._graph_loop = None

    monkeypatch.setattr(graph, "_graph", None)
    monkeypatch.setattr(graph, "_graph_loop", None)
    monkeypatch.setattr(graph.settings, "anthropic_api_key", "test-key", raising=False)
    monkeypatch.setattr(graph, "_get_checkpointer", fake_get_checkpointer)
    monkeypatch.setattr(graph, "_build_graph", lambda: builder)
    monkeypatch.setattr(graph, "cleanup_checkpointer", fake_cleanup)

    first = asyncio.run(graph._get_or_build_graph())
    second = asyncio.run(graph._get_or_build_graph())

    assert first == "compiled-1"
    assert second == "compiled-2"
    assert builder.compile_calls == ["checkpointer-1", "checkpointer-2"]


def test_cleanup_checkpointer_clears_cached_state(monkeypatch: MonkeyPatch) -> None:
    class FakeStack:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    stack = FakeStack()
    monkeypatch.setattr(graph, "_checkpointer_stack", stack)
    monkeypatch.setattr(graph, "_checkpointer", "checkpointer")
    monkeypatch.setattr(graph, "_checkpointer_loop", object())
    monkeypatch.setattr(graph, "_graph", "compiled-graph")
    monkeypatch.setattr(graph, "_graph_loop", object())

    asyncio.run(graph.cleanup_checkpointer())

    assert stack.closed is True
    assert graph._checkpointer_stack is None
    assert graph._checkpointer is None
    assert graph._checkpointer_loop is None
    assert graph._graph is None
    assert graph._graph_loop is None


def test_stream_agent_response_yields_only_text_chunks(monkeypatch: MonkeyPatch) -> None:
    fake_graph = FakeCompiledGraph(
        events=[
            {"event": "ignored"},
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": SimpleNamespace(
                        content=["Hello ", {"text": "there"}]
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content=None)},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="!")},
            },
        ]
    )

    async def fake_get_graph() -> FakeCompiledGraph:
        return fake_graph

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in graph.stream_agent_response("hi", "thread-1")
        ]

    monkeypatch.setattr(graph, "_get_or_build_graph", fake_get_graph)

    chunks = asyncio.run(collect())

    assert chunks == ["Hello there", "!"]
    assert fake_graph.stream_calls == [
        (
            {"messages": [("human", "hi")]},
            {"configurable": {"thread_id": "thread-1"}},
            "v2",
        )
    ]


def test_invoke_agent_returns_all_new_ai_messages(monkeypatch: MonkeyPatch) -> None:
    fake_graph = FakeCompiledGraph(
        invoke_result={
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="First"),
                AIMessage(content="Second"),
                AIMessage(content=[{"text": "Final response"}]),
            ]
        }
    )

    async def fake_get_graph() -> FakeCompiledGraph:
        return fake_graph

    monkeypatch.setattr(graph, "_get_or_build_graph", fake_get_graph)

    result = asyncio.run(graph.invoke_agent("hi", "thread-1"))

    assert result == "First\n\nSecond\n\nFinal response"
    assert fake_graph.invoke_calls == [
        (
            {"messages": [("human", "hi")]},
            {"configurable": {"thread_id": "thread-1"}},
        )
    ]
