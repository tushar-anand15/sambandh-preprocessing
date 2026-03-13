CREATE TABLE IF NOT EXISTS parse_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_id UUID NOT NULL UNIQUE,
    district_name TEXT,
    lb_name TEXT,
    lb_type TEXT,
    project_no TEXT,
    project_name TEXT,
    year_label TEXT,
    gcs_path TEXT NOT NULL,
    gcs_bucket TEXT NOT NULL,
    file_size_bytes INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    worker_id TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    parse_time_seconds FLOAT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    output_gcs_path TEXT,
    page_count INTEGER,
    block_count INTEGER,
    table_count INTEGER,
    output_size_bytes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_parse_jobs_status ON parse_jobs(status);
CREATE INDEX IF NOT EXISTS idx_parse_jobs_district ON parse_jobs(district_name);
CREATE INDEX IF NOT EXISTS idx_parse_jobs_completed ON parse_jobs(completed_at)
    WHERE status = 'done';

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    hostname TEXT,
    cpu_count INTEGER,
    cpu_percent FLOAT,
    mem_percent FLOAT,
    load_avg FLOAT,
    pdfs_completed INTEGER NOT NULL DEFAULT 0,
    pdfs_failed INTEGER NOT NULL DEFAULT 0,
    current_pdf TEXT,
    started_at TIMESTAMPTZ,
    last_seen TIMESTAMPTZ
);
