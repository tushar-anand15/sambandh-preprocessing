#!/usr/bin/env python3
"""
Standalone chunking script - processes completed OCR jobs.

Usage:
    python scripts/run_chunking.py [--batch-size 100] [--max-jobs 1000] [--dry-run]

Features:
    - Scans parse_jobs WHERE status='done' AND NOT IN chunk_jobs
    - Downloads parsed JSON from GCS
    - Runs chunker
    - Uploads to gs://parsed-sulekha/chunks/{path}
    - Logs progress to chunk_jobs table
    - Resumable: skips already-processed jobs
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import psycopg
from psycopg.rows import dict_row
from google.cloud import storage
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sambandh.chunker import Chunker, ChunkResult

console = Console()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://sambandh:sambandh@localhost:5433/sambandh"
)
GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET", "parsed-sulekha")

_gcs_client = None


def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def download_parsed_json(gcs_path: str) -> dict:
    """Download parsed JSON from GCS."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_OUTPUT_BUCKET)
    blob = bucket.blob(gcs_path)
    return json.loads(blob.download_as_text())


def upload_chunked_json(data: dict, output_path: str) -> int:
    """Upload chunked JSON to GCS. Returns size in bytes."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_OUTPUT_BUCKET)
    blob = bucket.blob(output_path)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    blob.upload_from_string(payload, content_type="application/json")
    return len(payload.encode("utf-8"))


def get_pending_jobs(conn, limit: int) -> list[dict]:
    """Get parse_jobs that are done but not yet chunked."""
    return conn.execute(
        """
        SELECT pj.id AS parse_job_id, pj.pdf_id, pj.output_gcs_path,
               pj.district_name, pj.lb_name, pj.lb_type, pj.project_no, pj.year_label
        FROM parse_jobs pj
        LEFT JOIN chunk_jobs cj ON pj.id = cj.parse_job_id
        WHERE pj.status = 'done'
          AND pj.output_gcs_path IS NOT NULL
          AND cj.id IS NULL
        ORDER BY pj.completed_at DESC
        LIMIT %s
        """,
        [limit],
    ).fetchall()


def get_existing_pending_jobs(conn, limit: int) -> list[dict]:
    """Get chunk_jobs already in the table with status='pending' (e.g. after --rechunk-all)."""
    return conn.execute(
        """
        SELECT cj.id, cj.parse_job_id, cj.pdf_id, cj.input_gcs_path,
               pj.district_name, pj.lb_name, pj.lb_type, pj.project_no, pj.year_label,
               pj.output_gcs_path
        FROM chunk_jobs cj
        JOIN parse_jobs pj ON cj.parse_job_id = pj.id
        WHERE cj.status = 'pending'
        ORDER BY cj.created_at
        LIMIT %s
        """,
        [limit],
    ).fetchall()


def get_failed_jobs(conn, limit: int) -> list[dict]:
    """Get chunk_jobs that failed and can be retried."""
    return conn.execute(
        """
        SELECT cj.id, cj.parse_job_id, cj.pdf_id, cj.input_gcs_path,
               pj.district_name, pj.lb_name, pj.lb_type, pj.project_no, pj.year_label
        FROM chunk_jobs cj
        JOIN parse_jobs pj ON cj.parse_job_id = pj.id
        WHERE cj.status = 'failed'
          AND cj.retry_count < 3
        ORDER BY cj.created_at
        LIMIT %s
        """,
        [limit],
    ).fetchall()


def insert_chunk_job(conn, parse_job_id: str, pdf_id: str, input_gcs_path: str) -> str:
    """Insert a new chunk_job and return its ID."""
    row = conn.execute(
        """
        INSERT INTO chunk_jobs (parse_job_id, pdf_id, input_gcs_path, status)
        VALUES (%s, %s, %s, 'pending')
        ON CONFLICT (parse_job_id) DO UPDATE SET status = chunk_jobs.status
        RETURNING id
        """,
        [parse_job_id, pdf_id, input_gcs_path],
    ).fetchone()
    return str(row["id"])


def update_chunk_job_started(conn, job_id: str):
    """Mark chunk job as started."""
    conn.execute(
        """
        UPDATE chunk_jobs
        SET status = 'started', started_at = %s
        WHERE id = %s
        """,
        [datetime.now(timezone.utc), job_id],
    )


def update_chunk_job_done(
    conn,
    job_id: str,
    chunk_time_seconds: float,
    chunk_count: int,
    kv_chunks: int,
    narrative_chunks: int,
    table_chunks: int,
    total_tokens: int,
    output_gcs_path: str,
):
    """Mark chunk job as done with stats."""
    conn.execute(
        """
        UPDATE chunk_jobs
        SET status = 'done',
            completed_at = %s,
            chunk_time_seconds = %s,
            chunk_count = %s,
            kv_chunks = %s,
            narrative_chunks = %s,
            table_chunks = %s,
            total_tokens = %s,
            output_gcs_path = %s
        WHERE id = %s
        """,
        [
            datetime.now(timezone.utc),
            chunk_time_seconds,
            chunk_count,
            kv_chunks,
            narrative_chunks,
            table_chunks,
            total_tokens,
            output_gcs_path,
            job_id,
        ],
    )


def update_chunk_job_failed(conn, job_id: str, error_message: str):
    """Mark chunk job as failed."""
    conn.execute(
        """
        UPDATE chunk_jobs
        SET status = 'failed',
            completed_at = %s,
            error_message = %s,
            retry_count = retry_count + 1
        WHERE id = %s
        """,
        [datetime.now(timezone.utc), error_message[:2000], job_id],
    )


def get_chunk_stats(conn) -> dict:
    """Get current chunking statistics."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'done') AS done,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed,
            COUNT(*) FILTER (WHERE status = 'pending') AS pending,
            COUNT(*) FILTER (WHERE status = 'started') AS started,
            COALESCE(SUM(chunk_count) FILTER (WHERE status = 'done'), 0) AS total_chunks,
            COALESCE(SUM(total_tokens) FILTER (WHERE status = 'done'), 0) AS total_tokens
        FROM chunk_jobs
        """
    ).fetchone()
    return dict(row)


def get_parse_jobs_done_count(conn) -> int:
    """Get count of completed parse jobs."""
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM parse_jobs WHERE status = 'done'"
    ).fetchone()
    return row["cnt"]


def process_job(
    conn, chunker: Chunker, job: dict, dry_run: bool = False
) -> tuple[bool, str]:
    """Process a single chunking job. Returns (success, message)."""
    input_path = job.get("output_gcs_path") or job.get("input_gcs_path")
    pdf_id = str(job["pdf_id"])
    parse_job_id = str(job["parse_job_id"])

    if dry_run:
        return True, f"Would process: {input_path}"

    chunk_job_id = job.get("id")
    if not chunk_job_id:
        chunk_job_id = insert_chunk_job(conn, parse_job_id, pdf_id, input_path)
        conn.commit()

    update_chunk_job_started(conn, chunk_job_id)
    conn.commit()

    start_time = time.time()

    try:
        doc = download_parsed_json(input_path)

        result: ChunkResult = chunker.chunk_document(doc)

        output_path = f"chunks/{input_path}"

        output_data = {
            "pdf_id": pdf_id,
            "source_path": input_path,
            "metadata": {
                "district": job.get("district_name"),
                "lb_name": job.get("lb_name"),
                "lb_type": job.get("lb_type"),
                "project_no": job.get("project_no"),
                "year": job.get("year_label"),
            },
            "chunk_count": result.stats.chunk_count,
            "chunks": [c.to_dict() for c in result.chunks],
            "stats": {
                "kv_chunks": result.stats.kv_chunks,
                "narrative_chunks": result.stats.narrative_chunks,
                "table_schema_chunks": result.stats.table_schema_chunks,
                "table_rows_chunks": result.stats.table_rows_chunks,
                "table_summary_chunks": result.stats.table_summary_chunks,
                "total_tokens": result.stats.total_tokens,
            },
        }

        upload_chunked_json(output_data, output_path)

        elapsed = time.time() - start_time

        table_chunks = (
            result.stats.table_schema_chunks
            + result.stats.table_rows_chunks
            + result.stats.table_summary_chunks
        )

        update_chunk_job_done(
            conn,
            chunk_job_id,
            chunk_time_seconds=elapsed,
            chunk_count=result.stats.chunk_count,
            kv_chunks=result.stats.kv_chunks,
            narrative_chunks=result.stats.narrative_chunks,
            table_chunks=table_chunks,
            total_tokens=result.stats.total_tokens,
            output_gcs_path=output_path,
        )
        conn.commit()

        return True, f"{result.stats.chunk_count} chunks in {elapsed:.1f}s"

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        update_chunk_job_failed(conn, chunk_job_id, error_msg)
        conn.commit()
        return False, error_msg


def print_stats(conn):
    """Print current statistics."""
    stats = get_chunk_stats(conn)
    parse_done = get_parse_jobs_done_count(conn)
    remaining = parse_done - stats["done"] - stats["failed"]

    table = Table(title="Chunking Progress")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Parse jobs done", f"{parse_done:,}")
    table.add_row("Chunk jobs done", f"{stats['done']:,}")
    table.add_row("Chunk jobs failed", f"{stats['failed']:,}")
    table.add_row("Remaining", f"{remaining:,}")
    table.add_row("Total chunks created", f"{stats['total_chunks']:,}")
    table.add_row("Total tokens", f"{stats['total_tokens']:,}")

    if stats["done"] > 0:
        avg_chunks = stats["total_chunks"] / stats["done"]
        avg_tokens = stats["total_tokens"] / stats["done"]
        table.add_row("Avg chunks/doc", f"{avg_chunks:.1f}")
        table.add_row("Avg tokens/doc", f"{avg_tokens:.0f}")

    console.print(table)


def reset_all_chunk_jobs(conn, max_jobs: int | None = None) -> int:
    """Reset done/failed chunk_jobs back to pending for re-chunking."""
    limit_clause = f"AND id IN (SELECT id FROM chunk_jobs WHERE status IN ('done', 'failed') LIMIT {max_jobs})" if max_jobs else ""
    row = conn.execute(
        f"""UPDATE chunk_jobs
            SET status = 'pending',
                started_at = NULL,
                completed_at = NULL,
                chunk_time_seconds = NULL,
                chunk_count = NULL,
                kv_chunks = NULL,
                narrative_chunks = NULL,
                table_chunks = NULL,
                total_tokens = NULL,
                output_gcs_path = NULL,
                error_message = NULL
            WHERE status IN ('done', 'failed')
            {limit_clause}
            RETURNING id"""
    ).fetchall()
    conn.commit()
    return len(row)


def main():
    parser = argparse.ArgumentParser(description="Run chunking on completed OCR jobs")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Jobs to process per batch"
    )
    parser.add_argument(
        "--max-jobs", type=int, default=None, help="Maximum jobs to process (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed"
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry failed jobs"
    )
    parser.add_argument(
        "--rechunk-all", action="store_true",
        help="Reset all done/failed chunk_jobs to pending and re-process from scratch"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show statistics and exit"
    )
    args = parser.parse_args()

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        if args.stats:
            print_stats(conn)
            return

        print_stats(conn)
        console.print()

        if args.rechunk_all:
            if args.dry_run:
                console.print("[yellow]DRY RUN - would reset all done/failed chunk jobs to pending[/yellow]")
                return
            reset_count = reset_all_chunk_jobs(conn, args.max_jobs)
            console.print(f"[yellow]Reset {reset_count} chunk jobs to pending[/yellow]")
            jobs = get_existing_pending_jobs(conn, args.max_jobs or 100000)
            console.print(f"[cyan]Found {len(jobs)} jobs to re-chunk[/cyan]")
        elif args.retry_failed:
            jobs = get_failed_jobs(conn, args.max_jobs or 1000)
            console.print(f"[yellow]Found {len(jobs)} failed jobs to retry[/yellow]")
        else:
            jobs = get_pending_jobs(conn, args.max_jobs or 100000)
            console.print(f"[cyan]Found {len(jobs)} pending jobs[/cyan]")

        if not jobs:
            console.print("[green]No jobs to process![/green]")
            return

        if args.dry_run:
            console.print(f"\n[yellow]DRY RUN - would process {len(jobs)} jobs:[/yellow]")
            for job in jobs[:10]:
                path = job.get("output_gcs_path") or job.get("input_gcs_path")
                console.print(f"  - {path}")
            if len(jobs) > 10:
                console.print(f"  ... and {len(jobs) - 10} more")
            return

        chunker = Chunker()
        success_count = 0
        fail_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking...", total=len(jobs))

            for i, job in enumerate(jobs):
                path = job.get("output_gcs_path") or job.get("input_gcs_path")
                short_path = path.split("/")[-1] if path else "unknown"

                progress.update(task, description=f"[cyan]{short_path}[/cyan]")

                success, message = process_job(conn, chunker, job)

                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    console.print(f"[red]Failed:[/red] {short_path}: {message}")

                progress.advance(task)

        console.print()
        console.print(f"[bold green]Completed: {success_count} succeeded, {fail_count} failed[/bold green]")
        console.print()
        print_stats(conn)


if __name__ == "__main__":
    main()
