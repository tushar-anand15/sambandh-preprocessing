#!/usr/bin/env python3
"""
Embedding pipeline - two-phase batch processing for maximum GPU utilization.

Phase 1: Load all chunks from GCS into processed_chunks (parallel downloads, single bulk insert)
Phase 2: Embed all chunks in one batch (GPU stays saturated), bulk insert embeddings

Usage:
    python scripts/run_embedding.py --device cuda --fp16
    python scripts/run_embedding.py --stats
    python scripts/run_embedding.py --load-only    # Phase 1 only
    python scripts/run_embedding.py --embed-only   # Phase 2 only
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

console = Console()
log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://sambandh:sambandh@localhost:5433/sambandh"
)
GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET", "parsed-sulekha")

BGE_M3_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_BATCH_SIZE_CPU = 16
EMBED_BATCH_SIZE_GPU = 512
EMBED_MAX_LENGTH = 8192
MIN_SEARCH_TEXT_LEN = 10

REQUIRED_CHUNK_KEYS = {"chunk_type", "search_text", "display_text", "page_start", "page_end"}
GCS_DOWNLOAD_WORKERS = 16

_gcs_client = None


def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def download_chunk_json(gcs_path: str) -> dict | None:
    """Download chunk JSON from GCS. Returns None on failure."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_OUTPUT_BUCKET)
        blob = bucket.blob(gcs_path)
        return json.loads(blob.download_as_text())
    except Exception as e:
        log.warning("Failed to download %s: %s", gcs_path, e)
        return None


def validate_chunk(chunk: dict, index: int) -> tuple[bool, str]:
    """Validate a single chunk dict. Returns (is_valid, reason)."""
    missing = REQUIRED_CHUNK_KEYS - chunk.keys()
    if missing:
        return False, f"chunk[{index}] missing keys: {missing}"
    return True, ""


def should_embed(search_text: str) -> bool:
    """Return True if the search_text is long enough to produce a meaningful embedding."""
    return len(search_text.strip()) >= MIN_SEARCH_TEXT_LEN


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_model = None


def load_embed_model(device: str = "cpu", fp16: bool = False):
    global _model
    if _model is not None:
        return _model

    console.print(f"[yellow]Loading BGE-M3 model on {device} (fp16={fp16})...[/yellow]")
    start = time.time()
    from FlagEmbedding import BGEM3FlagModel

    _model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=fp16, device=device)
    elapsed = time.time() - start
    console.print(f"[green]Model loaded in {elapsed:.1f}s[/green]\n")
    return _model


def embed_texts_batched(model, texts: list[str], batch_size: int, show_progress: bool = True) -> np.ndarray:
    """Embed all texts in a single model.encode() call. Returns float32 array (N, 1024)."""
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    console.print(f"  [dim]Calling model.encode() with {len(texts):,} texts, internal batch_size={batch_size}[/dim]")
    
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=EMBED_MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    
    return np.array(output["dense_vecs"], dtype=np.float32)


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------

def connect_db() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)
    return conn


def get_pending_jobs(conn) -> list[dict]:
    """Get chunk_jobs that are done but not yet in processed_chunks."""
    return conn.execute(
        """
        SELECT cj.id AS chunk_job_id, cj.pdf_id, cj.output_gcs_path,
               pj.district_name, pj.lb_name, pj.lb_type, pj.project_no, pj.year_label
        FROM chunk_jobs cj
        JOIN parse_jobs pj ON cj.parse_job_id = pj.id
        WHERE cj.status = 'done'
          AND cj.output_gcs_path IS NOT NULL
          AND cj.id NOT IN (SELECT DISTINCT chunk_job_id FROM processed_chunks)
        ORDER BY cj.completed_at DESC
        """
    ).fetchall()


def get_chunks_without_embeddings(conn) -> list[dict]:
    """Get all chunks that don't have embeddings yet."""
    return conn.execute(
        """
        SELECT pc.chunk_id, pc.search_text
        FROM processed_chunks pc
        LEFT JOIN chunk_embeddings ce ON pc.chunk_id = ce.chunk_id
        WHERE ce.id IS NULL
        ORDER BY pc.created_at
        """
    ).fetchall()


def bulk_insert_chunks(conn, rows: list[dict]):
    """Insert processed_chunks rows using COPY protocol."""
    if not rows:
        return

    columns = (
        "pdf_id", "chunk_job_id", "chunk_index", "chunk_type",
        "section_path", "display_text", "search_text",
        "page_start", "page_end", "token_count", "metadata",
        "district_name", "lb_name", "lb_type", "project_no", "year_label",
    )
    col_list = ", ".join(columns)

    with conn.cursor().copy(f"COPY processed_chunks ({col_list}) FROM STDIN") as copy:
        for r in rows:
            copy.write_row((
                r["pdf_id"],
                r["chunk_job_id"],
                r["chunk_index"],
                r["chunk_type"],
                r["section_path"],
                r["display_text"],
                r["search_text"],
                r["page_start"],
                r["page_end"],
                r["token_count"],
                json.dumps(r["metadata"], ensure_ascii=False),
                r["district_name"],
                r["lb_name"],
                r["lb_type"],
                r["project_no"],
                r["year_label"],
            ))


def bulk_insert_embeddings(conn, chunk_ids: list[str], vectors: np.ndarray, embed_model: str = "bge-m3-v1"):
    """Insert chunk_embeddings rows using COPY protocol."""
    if not chunk_ids:
        return

    with conn.cursor().copy(
        "COPY chunk_embeddings (chunk_id, embed_model, embedding) FROM STDIN"
    ) as copy:
        for chunk_id, vec in zip(chunk_ids, vectors):
            copy.write_row((chunk_id, embed_model, vec))


def get_stats(conn) -> dict:
    row = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM chunk_jobs WHERE status = 'done') AS chunk_jobs_done,
            (SELECT COUNT(DISTINCT chunk_job_id) FROM processed_chunks) AS jobs_loaded,
            (SELECT COUNT(*) FROM processed_chunks) AS total_chunks,
            (SELECT COUNT(*) FROM chunk_embeddings) AS total_embeddings
        """
    ).fetchone()
    return dict(row)


def print_stats(conn):
    stats = get_stats(conn)
    jobs_remaining = stats["chunk_jobs_done"] - stats["jobs_loaded"]
    chunks_without_embed = stats["total_chunks"] - stats["total_embeddings"]

    from rich.table import Table
    table = Table(title="Embedding Pipeline Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Chunk jobs done", f"{stats['chunk_jobs_done']:,}")
    table.add_row("Jobs loaded to DB", f"{stats['jobs_loaded']:,}")
    table.add_row("Jobs remaining (Phase 1)", f"{jobs_remaining:,}")
    table.add_row("Total chunks in DB", f"{stats['total_chunks']:,}")
    table.add_row("Chunks with embeddings", f"{stats['total_embeddings']:,}")
    table.add_row("Chunks needing embed (Phase 2)", f"{chunks_without_embed:,}")

    console.print(table)


# ---------------------------------------------------------------------------
# Phase 1: Load chunks from GCS into DB
# ---------------------------------------------------------------------------

def download_and_parse_job(job: dict) -> list[dict]:
    """Download JSON and parse into chunk rows. Returns empty list on failure."""
    gcs_path = job["output_gcs_path"]
    doc = download_chunk_json(gcs_path)
    if doc is None:
        return []

    chunks_raw = doc.get("chunks", [])
    if not chunks_raw:
        return []

    meta = doc.get("metadata", {})
    district = meta.get("district") or job.get("district_name")
    lb_name = meta.get("lb_name") or job.get("lb_name")
    lb_type = meta.get("lb_type") or job.get("lb_type")
    project_no = meta.get("project_no") or job.get("project_no")
    year_label = meta.get("year") or job.get("year_label")

    rows = []
    for i, chunk in enumerate(chunks_raw):
        valid, _ = validate_chunk(chunk, i)
        if not valid:
            continue

        rows.append({
            "pdf_id": str(job["pdf_id"]),
            "chunk_job_id": str(job["chunk_job_id"]),
            "chunk_index": i,
            "chunk_type": chunk["chunk_type"],
            "section_path": chunk.get("section_path", []),
            "display_text": chunk["display_text"],
            "search_text": chunk["search_text"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "token_count": chunk.get("metadata", {}).get("token_count"),
            "metadata": chunk.get("metadata", {}),
            "district_name": district,
            "lb_name": lb_name,
            "lb_type": lb_type,
            "project_no": project_no,
            "year_label": year_label,
        })

    return rows


def phase1_load_chunks(conn, jobs: list[dict]) -> int:
    """Download all chunk JSONs in parallel and bulk insert to DB. Returns chunk count."""
    console.print(f"\n[bold]Phase 1: Loading {len(jobs):,} jobs from GCS...[/bold]")

    all_rows = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading from GCS...", total=len(jobs))

        with ThreadPoolExecutor(max_workers=GCS_DOWNLOAD_WORKERS) as executor:
            futures = {executor.submit(download_and_parse_job, job): job for job in jobs}

            for future in as_completed(futures):
                rows = future.result()
                if rows:
                    all_rows.extend(rows)
                else:
                    failed += 1
                progress.advance(task)

    console.print(f"  Downloaded {len(jobs) - failed:,} jobs, {failed:,} failed")
    console.print(f"  Total chunks: {len(all_rows):,}")

    if all_rows:
        console.print("  Inserting into database...")
        start = time.time()
        bulk_insert_chunks(conn, all_rows)
        conn.commit()
        elapsed = time.time() - start
        console.print(f"  [green]Inserted {len(all_rows):,} chunks in {elapsed:.1f}s[/green]")

    return len(all_rows)


# ---------------------------------------------------------------------------
# Phase 2: Embed all chunks
# ---------------------------------------------------------------------------

def phase2_embed_chunks(conn, model, batch_size: int) -> int:
    """Embed all chunks that don't have embeddings yet. Returns embedding count."""
    chunks = get_chunks_without_embeddings(conn)
    if not chunks:
        console.print("\n[green]Phase 2: No chunks need embedding![/green]")
        return 0

    console.print(f"\n[bold]Phase 2: Embedding {len(chunks):,} chunks...[/bold]")

    # Filter to embeddable chunks
    embeddable = [(str(c["chunk_id"]), c["search_text"]) for c in chunks if should_embed(c["search_text"])]
    skipped = len(chunks) - len(embeddable)

    if skipped > 0:
        console.print(f"  Skipping {skipped:,} chunks with short search_text")

    if not embeddable:
        console.print("  [yellow]No chunks to embed after filtering[/yellow]")
        return 0

    chunk_ids = [cid for cid, _ in embeddable]
    texts = [text for _, text in embeddable]

    console.print(f"  Embedding {len(texts):,} texts (batch_size={batch_size})...")
    start = time.time()
    vectors = embed_texts_batched(model, texts, batch_size)
    embed_time = time.time() - start

    console.print(f"  Embedding took {embed_time:.1f}s ({len(texts) / embed_time:.0f} texts/sec)")

    console.print("  Inserting embeddings into database...")
    start = time.time()
    bulk_insert_embeddings(conn, chunk_ids, vectors)
    conn.commit()
    insert_time = time.time() - start
    console.print(f"  [green]Inserted {len(chunk_ids):,} embeddings in {insert_time:.1f}s[/green]")

    return len(chunk_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-phase embedding pipeline")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for embedding")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision (GPU only)")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size")
    parser.add_argument("--stats", action="store_true", help="Show statistics and exit")
    parser.add_argument("--load-only", action="store_true", help="Phase 1 only: load chunks from GCS")
    parser.add_argument("--embed-only", action="store_true", help="Phase 2 only: embed existing chunks")
    parser.add_argument("--reset", action="store_true", help="Clear processed_chunks and chunk_embeddings tables")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = EMBED_BATCH_SIZE_GPU if args.device == "cuda" else EMBED_BATCH_SIZE_CPU

    conn = connect_db()

    try:
        if args.reset:
            console.print("[yellow]Resetting tables...[/yellow]")
            conn.execute("TRUNCATE chunk_embeddings, processed_chunks CASCADE")
            conn.commit()
            console.print("[green]Tables cleared.[/green]")
            return

        if args.stats:
            print_stats(conn)
            return

        print_stats(conn)

        # Phase 1: Load chunks
        if not args.embed_only:
            jobs = get_pending_jobs(conn)
            if jobs:
                phase1_load_chunks(conn, jobs)
            else:
                console.print("\n[green]Phase 1: All jobs already loaded![/green]")

        # Phase 2: Embed chunks
        if not args.load_only:
            model = load_embed_model(device=args.device, fp16=args.fp16)
            phase2_embed_chunks(conn, model, args.batch_size)

        console.print()
        print_stats(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
