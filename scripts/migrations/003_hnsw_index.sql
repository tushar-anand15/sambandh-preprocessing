-- Migration: 003_hnsw_index
-- Description: Build HNSW index on chunk_embeddings for dense ANN search
-- Date: 2026-03-13
--
-- Run AFTER the backfill completes. Building on a full table is 10-50x faster
-- than incremental index maintenance during bulk inserts.
--
-- Usage: python scripts/run_migrations.py
--   (or run manually after run_embedding.py finishes)

CREATE INDEX IF NOT EXISTS idx_ce_hnsw ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
