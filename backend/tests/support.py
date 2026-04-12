from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class FakeResult:
    """Minimal query result wrapper with a ``data`` attribute."""

    data: Any


class FakeQuery:
    """Chainable query stub that records inputs and returns fixed data."""

    def __init__(self, data: Any = None) -> None:
        self.data = data
        self.operations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.insert_payloads: list[Any] = []
        self.update_payloads: list[Any] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> "FakeQuery":
        self.operations.append((name, args, kwargs))
        return self

    def select(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("select", *args, **kwargs)

    def eq(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("eq", *args, **kwargs)

    def ilike(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("ilike", *args, **kwargs)

    def order(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("order", *args, **kwargs)

    def neq(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("neq", *args, **kwargs)

    def gte(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("gte", *args, **kwargs)

    def lt(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("lt", *args, **kwargs)

    def gt(self, *args: Any, **kwargs: Any) -> "FakeQuery":
        return self._record("gt", *args, **kwargs)

    def insert(self, payload: Any) -> "FakeQuery":
        self.insert_payloads.append(payload)
        return self._record("insert", payload)

    def update(self, payload: Any) -> "FakeQuery":
        self.update_payloads.append(payload)
        return self._record("update", payload)

    def execute(self) -> FakeResult:
        self.operations.append(("execute", (), {}))
        return FakeResult(data=self.data)


class FakeSupabase:
    """Supabase stub with per-table/per-RPC query queues."""

    def __init__(
        self,
        *,
        tables: dict[str, list[FakeQuery]] | None = None,
        rpcs: dict[str, list[FakeQuery]] | None = None,
    ) -> None:
        self._tables: dict[str, deque[FakeQuery]] = {
            name: deque(queries)
            for name, queries in (tables or {}).items()
        }
        self._rpcs: dict[str, deque[FakeQuery]] = {
            name: deque(queries)
            for name, queries in (rpcs or {}).items()
        }
        self.table_calls: list[str] = []
        self.rpc_calls: list[tuple[str, dict[str, Any]]] = []

    def table(self, name: str) -> FakeQuery:
        self.table_calls.append(name)
        if name not in self._tables or not self._tables[name]:
            raise AssertionError(f"Unexpected table call: {name}")
        return self._tables[name].popleft()

    def rpc(self, name: str, params: dict[str, Any]) -> FakeQuery:
        self.rpc_calls.append((name, params))
        if name not in self._rpcs or not self._rpcs[name]:
            raise AssertionError(f"Unexpected rpc call: {name}")
        return self._rpcs[name].popleft()


class FakeAsyncAgent:
    """Async agent stub that returns a predefined payload."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return self.result


class FakeCompiledGraph:
    """Compiled graph stub for invoke/stream tests."""

    def __init__(
        self,
        *,
        events: list[dict[str, Any]] | None = None,
        invoke_result: dict[str, Any] | None = None,
    ) -> None:
        self.events = events or []
        self.invoke_result = invoke_result or {}
        self.stream_calls: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        self.invoke_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def astream_events(
        self,
        input_message: dict[str, Any],
        *,
        config: dict[str, Any],
        version: str,
    ) -> AsyncIterator[dict[str, Any]]:
        self.stream_calls.append((input_message, config, version))
        for event in self.events:
            yield event

    async def ainvoke(
        self,
        input_message: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.invoke_calls.append((input_message, config))
        return self.invoke_result
