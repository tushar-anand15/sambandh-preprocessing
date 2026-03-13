-- Migration: 001_chunk_jobs
-- Description: Create chunk_jobs table for tracking chunking progress
-- Date: 2026-03-13

CREATE TABLE IF NOT EXISTS chunk_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parse_job_id UUID NOT NULL REFERENCES parse_jobs(id),
    pdf_id UUID NOT NULL,
    input_gcs_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    chunk_time_seconds FLOAT,
    chunk_count INTEGER,
    kv_chunks INTEGER,
    narrative_chunks INTEGER,
    table_chunks INTEGER,
    total_tokens INTEGER,
    output_gcs_path TEXT,
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(parse_job_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_jobs_status ON chunk_jobs(status);
CREATE INDEX IF NOT EXISTS idx_chunk_jobs_pdf_id ON chunk_jobs(pdf_id);
