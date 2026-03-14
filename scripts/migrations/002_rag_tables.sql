-- Migration: 002_rag_tables
-- Description: Create processed_chunks + chunk_embeddings tables for hybrid search
-- Date: 2026-03-13
--
-- processed_chunks: FTS-ready table with denormalized metadata
-- chunk_embeddings: dense vectors separated from text for clean scans
-- HNSW index deferred to 003 (build after backfill for speed)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS processed_chunks (
    chunk_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_id          UUID        NOT NULL,
    chunk_job_id    UUID        NOT NULL REFERENCES chunk_jobs(id),
    chunk_index     INT         NOT NULL,
    chunk_type      TEXT        NOT NULL,
    section_path    TEXT[]      NOT NULL DEFAULT '{}',
    display_text    TEXT        NOT NULL,
    search_text     TEXT        NOT NULL,
    page_start      INT         NOT NULL,
    page_end        INT         NOT NULL,
    token_count     INT,
    metadata        JSONB       NOT NULL DEFAULT '{}',
    -- denormalized filters for query-time filtering
    district_name   TEXT,
    lb_name         TEXT,
    lb_type         TEXT,
    project_no      TEXT,
    year_label      TEXT,
    -- FTS: language-agnostic, works for Malayalam + English
    tsv             TSVECTOR    GENERATED ALWAYS AS (
                        to_tsvector('simple', search_text)
                    ) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(chunk_job_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_pc_pdf      ON processed_chunks(pdf_id);
CREATE INDEX IF NOT EXISTS idx_pc_tsv      ON processed_chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS idx_pc_trgm     ON processed_chunks USING gin(search_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_pc_district ON processed_chunks(district_name);
CREATE INDEX IF NOT EXISTS idx_pc_year     ON processed_chunks(year_label);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID        NOT NULL REFERENCES processed_chunks(chunk_id) ON DELETE CASCADE,
    embed_model     TEXT        NOT NULL DEFAULT 'bge-m3-v1',
    embedding       VECTOR(1024) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(chunk_id, embed_model)
);
