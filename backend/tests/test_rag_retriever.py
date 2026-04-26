from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pytest import MonkeyPatch

from app.services import rag_retriever
from tests.support import FakeQuery, FakeSupabase


def test_get_openai_api_key_requires_configuration(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(rag_retriever.settings, "OPENAI_API_KEY", "", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not configured"):
        rag_retriever._get_openai_api_key()


def test_embed_texts_sorts_embeddings_by_index(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> SimpleNamespace:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "data": [
                    {"index": 1, "embedding": [2.0, 2.1]},
                    {"index": 0, "embedding": [1.0, 1.1]},
                ]
            },
        )

    monkeypatch.setattr(
        rag_retriever.settings, "OPENAI_API_KEY", "test-key", raising=False
    )
    monkeypatch.setattr(rag_retriever.httpx, "post", fake_post)

    result = rag_retriever.embed_texts(["first", "second"])

    assert captured["url"] == rag_retriever.OPENAI_EMBEDDING_URL
    assert captured["timeout"] == 60.0
    assert captured["json"] == {
        "model": rag_retriever.EMBEDDING_MODEL,
        "input": ["first", "second"],
        "dimensions": rag_retriever.EMBEDDING_DIMENSIONS,
    }
    assert result == [[1.0, 1.1], [2.0, 2.1]]


def test_embed_texts_raises_on_http_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        rag_retriever.settings, "OPENAI_API_KEY", "test-key", raising=False
    )
    monkeypatch.setattr(
        rag_retriever.httpx,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=500,
            text="upstream failure",
        ),
    )

    with pytest.raises(RuntimeError, match="OpenAI embedding API error"):
        rag_retriever.embed_texts(["query"])


def test_retrieve_medical_knowledge_parses_rpc_rows(monkeypatch: MonkeyPatch) -> None:
    fake_supabase = FakeSupabase(
        rpcs={
            "match_medical_knowledge": [
                FakeQuery(
                    [
                        {
                            "id": "chunk-1",
                            "content": "Cardiology content",
                            "metadata": json.dumps(
                                {
                                    "specialty_id": "spec-cardio",
                                    "specialty_name": "Cardiology",
                                }
                            ),
                            "similarity": 0.88,
                        }
                    ]
                )
            ]
        }
    )
    monkeypatch.setattr(rag_retriever, "supabase", fake_supabase)
    monkeypatch.setattr(
        rag_retriever,
        "embed_query",
        lambda query: [0.1, 0.2, 0.3],
    )

    result = rag_retriever.retrieve_medical_knowledge(
        "chest pain",
        match_count=3,
        match_threshold=0.4,
    )

    assert fake_supabase.rpc_calls == [
        (
            "match_medical_knowledge",
            {
                "query_embedding": [0.1, 0.2, 0.3],
                "match_count": 3,
                "match_threshold": 0.4,
            },
        )
    ]
    assert result == [
        {
            "id": "chunk-1",
            "content": "Cardiology content",
            "metadata": {
                "specialty_id": "spec-cardio",
                "specialty_name": "Cardiology",
            },
            "similarity": 0.88,
        }
    ]


def test_retrieve_medical_knowledge_returns_empty_list(
    monkeypatch: MonkeyPatch,
) -> None:
    fake_supabase = FakeSupabase(rpcs={"match_medical_knowledge": [FakeQuery([])]})
    monkeypatch.setattr(rag_retriever, "supabase", fake_supabase)
    monkeypatch.setattr(rag_retriever, "embed_query", lambda query: [0.1, 0.2])

    result = rag_retriever.retrieve_medical_knowledge("no results expected")

    assert result == []
