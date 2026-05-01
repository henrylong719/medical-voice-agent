"""
Ingest medical knowledge chunks into the medical_knowledge table.

This script:
  1. Reads the chunk definitions from knowledge_chunks.py
  2. Embeds each chunk's content using OpenAI text-embedding-3-small
  3. Upserts the chunk + embedding + metadata into medical_knowledge

Run from the backend/ directory:
    python -m app.services.ingest_knowledge

Idempotent: safe to re-run. Uses the md5(content) unique index to
detect duplicates — unchanged chunks are skipped, new/modified chunks
are inserted.

Requires OPENAI_API_KEY in backend/.env (not needed anywhere else
in this project — we use it only for embeddings).
"""

from __future__ import annotations

import json
import sys
import time
from typing import cast

from postgrest.types import CountMethod

from app.core.config import settings
from app.supabase_client import supabase
from app.services.knowledge_chunks import KNOWLEDGE_CHUNKS
from app.services.rag_retriever import embed_texts


# ============================================================
# HELPERS
# ============================================================


def _get_openai_api_key() -> str:
    """Validate that the OpenAI API key is configured."""
    key = settings.OPENAI_API_KEY.strip()
    if not key:
        print(
            "ERROR: OPENAI_API_KEY not found in backend/.env\n"
            "Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "You can get one at https://platform.openai.com/api-keys"
        )
        sys.exit(1)
    return key


# ============================================================
# INGESTION
# ============================================================


def ingest_chunks() -> None:
    """
    Embed and upsert all knowledge chunks into medical_knowledge.

    For each chunk:
      1. Check if it already exists (by md5 of content)
      2. If not, insert the content + metadata
      3. Embed and update the embedding column

    We separate insert and embedding update so that if embedding
    fails partway through, the content is still in the DB and
    we can retry just the embedding step.
    """
    # Fail fast if the API key is missing — before we do any DB work
    _get_openai_api_key()

    print(f"Found {len(KNOWLEDGE_CHUNKS)} knowledge chunks to process\n")

    # ── Step 1: Insert or skip chunks ────────────────────────
    chunks_to_embed: list[dict] = []

    for i, chunk in enumerate(KNOWLEDGE_CHUNKS, 1):
        content = chunk["content"]
        metadata = chunk["metadata"]

        # Check if this exact content already exists
        existing = (
            supabase.table("medical_knowledge")
            .select("id, embedding")
            .eq("content", content)
            .execute()
        )

        if existing.data:
            row = cast(dict, existing.data[0])
            if row.get("embedding"):
                print(
                    f"  [{i}/{len(KNOWLEDGE_CHUNKS)}] Skip (already embedded): "
                    f"{content[:60]}..."
                )
                continue
            else:
                # Content exists but embedding is missing — need to embed
                print(
                    f"  [{i}/{len(KNOWLEDGE_CHUNKS)}] Exists, needs embedding: "
                    f"{content[:60]}..."
                )
                chunks_to_embed.append(
                    {
                        "id": row["id"],
                        "content": content,
                    }
                )
                continue

        # Insert new chunk (without embedding — we'll batch-embed below)
        result = (
            supabase.table("medical_knowledge")
            .insert(
                {
                    "content": content,
                    "metadata": json.dumps(metadata)
                    if isinstance(metadata, dict)
                    else metadata,
                }
            )
            .execute()
        )

        if result.data:
            row_id = cast(dict, result.data[0])["id"]
            print(f"  [{i}/{len(KNOWLEDGE_CHUNKS)}] Inserted: {content[:60]}...")
            chunks_to_embed.append(
                {
                    "id": row_id,
                    "content": content,
                }
            )
        else:
            print(
                f"  [{i}/{len(KNOWLEDGE_CHUNKS)}] FAILED to insert: {content[:60]}..."
            )

    # ── Step 2: Batch embed ──────────────────────────────────
    if not chunks_to_embed:
        print("\nAll chunks already embedded. Nothing to do!")
        return

    print(f"\nEmbedding {len(chunks_to_embed)} chunks...")
    start = time.time()

    texts = [c["content"] for c in chunks_to_embed]
    embeddings = embed_texts(texts)

    elapsed = time.time() - start
    print(f"Embedding complete in {elapsed:.1f}s")

    # ── Step 3: Update embeddings ────────────────────────────
    print(f"\nSaving embeddings to database...")

    for chunk_info, embedding in zip(chunks_to_embed, embeddings):
        # pgvector expects the embedding as a JSON array string
        supabase.table("medical_knowledge").update(
            {
                "embedding": embedding,
            }
        ).eq("id", chunk_info["id"]).execute()

    print(f"\nDone! {len(chunks_to_embed)} chunks embedded and saved.")

    # ── Step 4: Verify ───────────────────────────────────────
    total = (
        supabase.table("medical_knowledge")
        .select("id", count=CountMethod.exact)
        .execute()
    )
    embedded = (
        supabase.table("medical_knowledge")
        .select("id", count=CountMethod.exact)
        .not_.is_("embedding", "null")
        .execute()
    )

    print(f"\nVerification:")
    print(f"  Total chunks in DB:    {total.count}")
    print(f"  Chunks with embedding: {embedded.count}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Medical Knowledge Ingestion Pipeline")
    print("=" * 60)
    print()
    ingest_chunks()
