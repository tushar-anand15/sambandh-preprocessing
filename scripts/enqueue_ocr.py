#!/usr/bin/env python3
"""Enqueue PDFs for OCR processing.

Reads PDF metadata from the scraper Postgres DB (read-only) and inserts
rows into the sambandh parse_jobs table. Idempotent: re-running with the
same filters skips already-enqueued PDFs via ON CONFLICT DO NOTHING.

Only newly inserted rows get Celery tasks submitted.
"""

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()

SCRAPER_DB_URL = os.environ.get(
    "SCRAPER_DATABASE_URL", "postgresql+psycopg://sulekha:sulekha@localhost:5432/sulekha"
)
SAMBANDH_DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://sambandh:sambandh@localhost:5433/sambandh"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/0")


def _clean_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


def fetch_pdfs_from_scraper(year_label: str, district: str | None, limit: int | None) -> list[dict]:
    sql = """
    SELECT
        p.id AS pdf_id,
        p.gcs_path,
        p.gcs_bucket,
        p.file_size_bytes,
        pr.project_no,
        pr.project_name,
        lb.lb_name,
        d.district_name,
        d.lb_type_label AS lb_type,
        d.year_label
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
      AND d.year_label = %s
    """
    params: list = [year_label]

    if district:
        sql += " AND d.district_name = %s"
        params.append(district)

    sql += " ORDER BY d.district_name, lb.lb_name, pr.project_no"

    if limit:
        sql += f" LIMIT {int(limit)}"

    with psycopg.connect(_clean_url(SCRAPER_DB_URL)) as conn:
        rows = conn.execute(sql, params).fetchall()
        cols = [
            "pdf_id", "gcs_path", "gcs_bucket", "file_size_bytes",
            "project_no", "project_name", "lb_name", "district_name",
            "lb_type", "year_label",
        ]
        return [dict(zip(cols, row)) for row in rows]


def insert_jobs(pdfs: list[dict]) -> list[str]:
    """Insert into parse_jobs, returning IDs of newly created rows."""
    new_ids = []
    with psycopg.connect(_clean_url(SAMBANDH_DB_URL)) as conn:
        for pdf in pdfs:
            row = conn.execute(
                """INSERT INTO parse_jobs
                       (pdf_id, district_name, lb_name, lb_type, project_no,
                        project_name, year_label, gcs_path, gcs_bucket, file_size_bytes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (pdf_id) DO NOTHING
                   RETURNING id""",
                [
                    str(pdf["pdf_id"]),
                    pdf["district_name"],
                    pdf["lb_name"],
                    pdf["lb_type"],
                    str(pdf["project_no"]),
                    pdf["project_name"],
                    pdf["year_label"],
                    pdf["gcs_path"],
                    pdf["gcs_bucket"],
                    pdf["file_size_bytes"],
                ],
            ).fetchone()
            if row:
                new_ids.append(str(row[0]))
        conn.commit()
    return new_ids


def submit_celery_tasks(job_ids: list[str]):
    os.environ.setdefault("REDIS_URL", REDIS_URL)
    from sambandh.tasks.ocr import parse_pdf

    for job_id in job_ids:
        parse_pdf.delay(job_id)


def get_pending_job_ids(year_label: str, district: str | None, limit: int | None) -> list[str]:
    """Get job IDs for pending jobs that need to be re-queued."""
    sql = "SELECT id FROM parse_jobs WHERE status = 'pending' AND year_label = %s"
    params: list = [year_label]

    if district:
        sql += " AND district_name = %s"
        params.append(district)

    sql += " ORDER BY created_at"

    if limit:
        sql += f" LIMIT {int(limit)}"

    with psycopg.connect(_clean_url(SAMBANDH_DB_URL)) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [str(row[0]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Enqueue PDFs for OCR processing")
    parser.add_argument("--year", required=True, help="Year label, e.g. 2025-2026")
    parser.add_argument("--district", help="Filter by district name")
    parser.add_argument("--limit", type=int, help="Max PDFs to enqueue (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Count without inserting")
    parser.add_argument("--requeue-pending", action="store_true", help="Re-submit Celery tasks for pending jobs")
    args = parser.parse_args()

    console.print(f"[bold]Enqueue OCR jobs[/bold]")
    console.print(f"  Year: {args.year}")
    console.print(f"  District: {args.district or 'ALL'}")
    console.print(f"  Limit: {args.limit or 'none'}\n")

    console.print("[yellow]Querying scraper DB...[/yellow]")
    pdfs = fetch_pdfs_from_scraper(args.year, args.district, args.limit)
    console.print(f"  Found {len(pdfs)} PDFs in scraper DB\n")

    if not pdfs:
        console.print("[red]No PDFs found matching filter[/red]")
        sys.exit(1)

    if args.dry_run:
        console.print("[yellow]Dry run -- not inserting[/yellow]")
        return

    console.print("[yellow]Inserting into parse_jobs...[/yellow]")
    new_ids = insert_jobs(pdfs)
    skipped = len(pdfs) - len(new_ids)
    console.print(f"  [green]{len(new_ids)} new jobs created[/green]")
    console.print(f"  [dim]{skipped} already existed (skipped)[/dim]\n")

    if args.requeue_pending:
        console.print("[yellow]Finding pending jobs to re-queue...[/yellow]")
        pending_ids = get_pending_job_ids(args.year, args.district, args.limit)
        if pending_ids:
            console.print(f"  Found {len(pending_ids)} pending jobs")
            console.print("[yellow]Submitting Celery tasks...[/yellow]")
            submit_celery_tasks(pending_ids)
            console.print(f"  [bold green]{len(pending_ids)} tasks re-submitted to Redis[/bold green]")
        else:
            console.print("  [dim]No pending jobs found[/dim]")
        return

    if not new_ids:
        console.print("[green]Nothing new to enqueue[/green]")
        return

    console.print("[yellow]Submitting Celery tasks...[/yellow]")
    submit_celery_tasks(new_ids)
    console.print(f"  [bold green]{len(new_ids)} tasks submitted to Redis[/bold green]")


if __name__ == "__main__":
    main()
