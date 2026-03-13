"""Postgres connection pool and query helpers for parse_jobs / worker_heartbeats."""

import os
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://sambandh:sambandh@localhost:5433/sambandh"
)

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"),
            min_size=1,
            max_size=3,
            kwargs={"row_factory": dict_row},
        )
    return _pool


@contextmanager
def get_conn():
    pool = _get_pool()
    with pool.connection() as conn:
        yield conn


def get_job(job_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM parse_jobs WHERE id = %s", [job_id]
        ).fetchone()
    return row


def update_job_started(job_id: str, worker_id: str):
    with get_conn() as conn:
        conn.execute(
            """UPDATE parse_jobs
               SET status = 'started', worker_id = %s, started_at = %s
               WHERE id = %s AND status IN ('pending', 'started')""",
            [worker_id, datetime.now(timezone.utc), job_id],
        )


def update_job_done(
    job_id: str,
    parse_time_seconds: float,
    output_gcs_path: str,
    page_count: int,
    block_count: int,
    table_count: int,
    output_size_bytes: int,
):
    with get_conn() as conn:
        conn.execute(
            """UPDATE parse_jobs
               SET status = 'done',
                   completed_at = %s,
                   parse_time_seconds = %s,
                   output_gcs_path = %s,
                   page_count = %s,
                   block_count = %s,
                   table_count = %s,
                   output_size_bytes = %s
               WHERE id = %s""",
            [
                datetime.now(timezone.utc),
                parse_time_seconds,
                output_gcs_path,
                page_count,
                block_count,
                table_count,
                output_size_bytes,
                job_id,
            ],
        )


def update_job_failed(job_id: str, error_message: str, *, retry: bool = False):
    status = "pending" if retry else "failed"
    with get_conn() as conn:
        conn.execute(
            """UPDATE parse_jobs
               SET status = %s,
                   completed_at = %s,
                   error_message = %s,
                   retry_count = retry_count + 1
               WHERE id = %s""",
            [status, datetime.now(timezone.utc), error_message[:2000], job_id],
        )


def upsert_heartbeat(
    worker_id: str,
    hostname: str,
    cpu_count: int,
    cpu_percent: float,
    mem_percent: float,
    load_avg: float,
    pdfs_completed: int,
    pdfs_failed: int,
    current_pdf: str | None,
    machine_name: str | None = None,
):
    now = datetime.now(timezone.utc)
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO worker_heartbeats
                   (worker_id, hostname, machine_name, cpu_count, cpu_percent, mem_percent,
                    load_avg, pdfs_completed, pdfs_failed, current_pdf, started_at, last_seen)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (worker_id) DO UPDATE SET
                   machine_name = EXCLUDED.machine_name,
                   cpu_percent = EXCLUDED.cpu_percent,
                   mem_percent = EXCLUDED.mem_percent,
                   load_avg = EXCLUDED.load_avg,
                   pdfs_completed = EXCLUDED.pdfs_completed,
                   pdfs_failed = EXCLUDED.pdfs_failed,
                   current_pdf = EXCLUDED.current_pdf,
                   last_seen = EXCLUDED.last_seen""",
            [
                worker_id,
                hostname,
                machine_name,
                cpu_count,
                cpu_percent,
                mem_percent,
                load_avg,
                pdfs_completed,
                pdfs_failed,
                current_pdf,
                now,
                now,
            ],
        )


def reset_stuck_jobs(timeout_minutes: int = 20) -> list[dict]:
    """Reset jobs stuck in 'started' state back to 'pending'."""
    with get_conn() as conn:
        rows = conn.execute(
            """UPDATE parse_jobs 
               SET status = 'pending', worker_id = NULL, started_at = NULL 
               WHERE status = 'started' 
                 AND started_at < now() - interval '%s minutes'
               RETURNING id, gcs_path, worker_id""",
            [timeout_minutes],
        ).fetchall()
    return rows


def delete_stale_heartbeats(timeout_minutes: int = 10) -> list[dict]:
    """Remove worker heartbeats that haven't been updated recently."""
    with get_conn() as conn:
        rows = conn.execute(
            """DELETE FROM worker_heartbeats 
               WHERE last_seen < now() - interval '%s minutes'
               RETURNING worker_id, last_seen""",
            [timeout_minutes],
        ).fetchall()
    return rows


def get_active_heartbeats() -> list[dict]:
    """Get all worker heartbeats for snapshotting."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT worker_id, hostname, machine_name, cpu_percent, mem_percent, 
                      load_avg, pdfs_completed, pdfs_failed
               FROM worker_heartbeats 
               WHERE last_seen > now() - interval '5 minutes'"""
        ).fetchall()
    return rows


def insert_metrics_snapshot(
    worker_id: str,
    hostname: str | None,
    machine_name: str,
    cpu_percent: float,
    mem_percent: float,
    load_avg: float,
    pdfs_completed_total: int,
    pdfs_in_interval: int,
):
    """Insert a point-in-time metrics snapshot."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO worker_metrics_history
                   (worker_id, hostname, machine_name, cpu_percent, mem_percent,
                    load_avg, pdfs_completed_total, pdfs_in_interval)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                worker_id,
                hostname,
                machine_name,
                cpu_percent,
                mem_percent,
                load_avg,
                pdfs_completed_total,
                pdfs_in_interval,
            ],
        )


def insert_throughput_snapshot(
    interval_start: datetime,
    interval_end: datetime,
    machine_name: str | None,
    pdfs_completed: int,
    avg_parse_time: float | None,
):
    """Insert a throughput snapshot for a time interval."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO throughput_snapshots
                   (interval_start, interval_end, machine_name, pdfs_completed, avg_parse_time)
               VALUES (%s, %s, %s, %s, %s)""",
            [interval_start, interval_end, machine_name, pdfs_completed, avg_parse_time],
        )


def get_throughput_by_machine(since_minutes: int = 5) -> list[dict]:
    """Get completed PDFs grouped by machine for the last N minutes."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT 
                   COALESCE(
                       (SELECT machine_name FROM worker_heartbeats WHERE worker_heartbeats.worker_id = parse_jobs.worker_id),
                       SPLIT_PART(worker_id, '-', 2)
                   ) AS machine_name,
                   COUNT(*) AS pdfs_completed,
                   AVG(parse_time_seconds) AS avg_parse_time
               FROM parse_jobs 
               WHERE status = 'done' 
                 AND completed_at > now() - interval '%s minutes'
               GROUP BY 1""",
            [since_minutes],
        ).fetchall()
    return rows
