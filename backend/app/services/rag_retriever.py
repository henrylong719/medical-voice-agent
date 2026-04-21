"""
RAG retriever for medical knowledge.

Handles the semantic search side of hybrid triage:
  1. Embed the patient's symptom description using OpenAI
  2. Call the match_medical_knowledge RPC for similarity search
  3. Return ranked results with content, metadata, and scores

This module is used by the triage tool in agent/tools.py.
The keyword search side still uses symptom_specialty_map directly.

Why a separate module?
  - Keeps embedding logic out of the tool layer
  - Makes the retriever testable independently
  - Can be reused by other tools or endpoints later
"""

from __future__ import annotations

import json
from typing import Any, TypedDict, cast

import httpx

from app.config import settings
from app.supabase_client import supabase


# ============================================================
# CONFIGURATION
# ============================================================
# Must match what we used in the ingestion script — same model,
# same dimensions. If these don't match, the similarity search
# will return garbage because you'd be comparing vectors from
# different embedding spaces.
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"


# ============================================================
# TYPES
# ============================================================


class RetrievedChunk(TypedDict):
    """Shape of a result from the similarity search."""

    id: str
    content: str
    metadata: dict[str, Any]
    similarity: float


# ============================================================
# EMBEDDING
# ============================================================
# All embedding logic lives here. The ingestion script imports
# embed_texts() from this module rather than defining its own.
# This ensures both ingestion and retrieval always use the same
# model, dimensions, and API call pattern.
# ============================================================


def _get_openai_api_key() -> str:
    """Read and validate the OpenAI API key from settings."""
    api_key = settings.openai_api_key.strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it in backend/.env.")
    return api_key


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using OpenAI text-embedding-3-small.

    Sends all texts in a single API request for efficiency.
    OpenAI supports up to 2048 texts per batch.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors in the same order as input.

    Raises:
        RuntimeError: If the API key is missing or the API call fails.
    """
    api_key = _get_openai_api_key()

    response = httpx.post(
        OPENAI_EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
            "dimensions": EMBEDDING_DIMENSIONS,
        },
        timeout=60.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI embedding API error ({response.status_code}): {response.text}"
        )

    data = response.json()
    # Sort by index to match input order (OpenAI may return out of order)
    embedding_objects = sorted(data["data"], key=lambda x: x["index"])
    return [obj["embedding"] for obj in embedding_objects]


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string.

    Convenience wrapper around embed_texts() for the common case
    of embedding one patient query at retrieval time.

    Args:
        text: The patient's symptom description.

    Returns:
        1536-dimensional embedding vector.
    """
    return embed_texts([text])[0]


# ============================================================
# RETRIEVAL
# ============================================================


def retrieve_medical_knowledge(
    query: str,
    match_count: int = 5,
    match_threshold: float = 0.3,
) -> list[RetrievedChunk]:
    """
    Semantic search over the medical knowledge base.

    Embeds the query, then calls the match_medical_knowledge RPC
    function in Supabase to find the most similar chunks.

    Args:
        query: Natural language symptom description from the patient.
            Works best with full sentences: "I've been getting sharp
            pains behind my eyes with flashing lights" rather than
            just keywords: "eye pain."
        match_count: Maximum number of chunks to return (default 5).
        match_threshold: Minimum similarity score 0–1 (default 0.3).
            Lower = more results but possibly less relevant.

    Returns:
        List of RetrievedChunk dicts, sorted by similarity (highest first).
        Each contains: id, content, metadata (specialty_id, category, etc.),
        and similarity score.
    """
    # Step 1: Embed the patient's query
    query_embedding = embed_query(query)

    # Step 2: Call the Supabase RPC function
    result = supabase.rpc(
        "match_medical_knowledge",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "match_threshold": match_threshold,
        },
    ).execute()

    if not result.data:
        return []

    # Step 3: Parse and return results
    rows = cast(list[dict[str, Any]], result.data)
    chunks: list[RetrievedChunk] = []
    for row in rows:
        # metadata may come back as a JSON string or dict depending
        # on the Supabase client version
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        chunks.append(
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": metadata,
                "similarity": row["similarity"],
            }
        )

    return chunks
