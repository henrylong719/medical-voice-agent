-- ============================================================
-- Medical Voice Agent — Phase 3: RAG Migration
-- Enables pgvector, creates medical_knowledge table, and adds
-- the similarity search RPC function.
--
-- Run this in Supabase SQL Editor AFTER 001_schema.sql and 002_seed.sql.
-- ============================================================


-- ============================================================
-- 1. ENABLE PGVECTOR
-- ============================================================
-- This adds the `vector` data type, distance operators (<=> for
-- cosine, <-> for L2, <#> for inner product), and index types
-- (HNSW, IVFFlat) to your Postgres instance.
--
-- Supabase includes pgvector out of the box — we just need to
-- activate the extension.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 2. MEDICAL KNOWLEDGE TABLE
-- ============================================================
-- Each row is one "chunk" of medical knowledge that gets embedded.
-- The content column holds the rich text passage.
-- The embedding column holds the 1536-dim vector from
-- text-embedding-3-small.
--
-- Metadata (JSONB) stores structured info for filtering and
-- display without rigid column requirements:
--   - specialty_id: UUID linking to the specialties table
--   - specialty_name: denormalized for convenience in results
--   - category: what type of chunk ("symptom_cluster",
--     "severity_guide", "follow_up_questions")
--   - severity_keywords: words suggesting urgency
--   - source: where this knowledge came from
-- ============================================================

CREATE TABLE medical_knowledge (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content     TEXT NOT NULL,                  -- rich text passage (what gets embedded)
    embedding   vector(1536),                   -- populated by the ingestion script
    metadata    JSONB NOT NULL DEFAULT '{}',    -- specialty_id, category, severity, etc.
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Prevent duplicate chunks (same content shouldn't appear twice)
CREATE UNIQUE INDEX idx_medical_knowledge_content
    ON medical_knowledge (md5(content));

-- ============================================================
-- 3. HNSW INDEX ON EMBEDDINGS
-- ============================================================
-- This index makes similarity search fast by building a
-- multi-layer navigable graph over the vectors.
--
-- cosine_ops: use cosine distance (<=> operator)
-- m = 16: each node connects to 16 neighbors (default, good
--   balance of speed vs quality)
-- ef_construction = 64: how many candidates to consider when
--   building the graph (higher = better recall, slower build)
--
-- For our ~30 chunks this index is technically unnecessary —
-- Postgres would just brute-force scan them all in microseconds.
-- But we add it now because:
--   (a) it teaches you the pattern you'd use at scale
--   (b) it won't hurt performance at small scale
--   (c) you won't forget to add it later when it matters
-- ============================================================

CREATE INDEX idx_medical_knowledge_embedding
    ON medical_knowledge
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ============================================================
-- 4. SIMILARITY SEARCH RPC FUNCTION
-- ============================================================
-- This is the function your Python code will call via
-- supabase.rpc("match_medical_knowledge", {...}).
--
-- How it works:
--   1. Takes the query embedding (from your Python code)
--   2. Computes cosine distance to every chunk's embedding
--   3. Filters out anything below the similarity threshold
--   4. Returns the top-K most similar chunks
--
-- Returns: id, content, metadata, and similarity score (1 - distance).
-- Cosine distance ranges 0 (identical) to 2 (opposite),
-- so similarity = 1 - distance ranges -1 to 1.
--
-- Why an RPC function instead of a raw query?
--   - Supabase client calls it cleanly: supabase.rpc(...)
--   - Keeps vector math in the DB where it's optimized
--   - Easy to tune threshold/count without code changes
-- ============================================================

CREATE OR REPLACE FUNCTION match_medical_knowledge(
    query_embedding  vector(1536),        -- the embedded patient query
    match_count      int DEFAULT 5,       -- how many results to return
    match_threshold  float DEFAULT 0.3    -- minimum similarity (0–1 scale)
)
RETURNS TABLE (
    id          UUID,
    content     TEXT,
    metadata    JSONB,
    similarity  float
)
LANGUAGE sql STABLE                       -- STABLE = no side effects, safe to cache
AS $$
    SELECT
        mk.id,
        mk.content,
        mk.metadata,
        1 - (mk.embedding <=> query_embedding) AS similarity
    FROM medical_knowledge mk
    WHERE mk.embedding IS NOT NULL
      AND 1 - (mk.embedding <=> query_embedding) > match_threshold
    ORDER BY mk.embedding <=> query_embedding  -- ascending distance = descending similarity
    LIMIT match_count;
$$;
