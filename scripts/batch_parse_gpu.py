#!/usr/bin/env python3
"""
batch_parse_gpu.py — GPU-optimized batch PDF parsing with Docling.

Designed for running on GCP GPU instances (T4/L4/A100) to process
the full Sulekha PDF corpus efficiently.

Features:
- Checkpointing: Tracks processed PDFs in a state file, resumes on restart
- Batch processing: Configurable batch sizes for GPU memory optimization
- GCS integration: Downloads from GCS, uploads parsed JSON to GCS
- Progress tracking: Rich progress bars and ETA estimates
- Error handling: Quarantines failed PDFs, continues processing

Usage:
    # Process all 2025-2026 PDFs
    python scripts/batch_parse_gpu.py --year 2025-2026

    # Process specific district
    python scripts/batch_parse_gpu.py --year 2025-2026 --district Ernakulam

    # Limit batch size (for testing)
    python scripts/batch_parse_gpu.py --year 2025-2026 --limit 100

    # Use higher batch sizes on GPU with more VRAM
    python scripts/batch_parse_gpu.py --year 2025-2026 --layout-batch 64 --table-batch 8

Environment Variables:
    SCRAPER_DATABASE_URL: Postgres connection string for scraper DB
    GOOGLE_APPLICATION_CREDENTIALS: Path to GCS service account key
    GCS_BUCKET_NAME: GCS bucket for PDFs (default: sulekha-pdfs)
    GCS_OUTPUT_BUCKET: GCS bucket for parsed output (default: same as input)
    OUTPUT_PREFIX: GCS prefix for output files (default: parsed/)
"""

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from google.cloud import storage
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

load_dotenv()

console = Console()

DATABASE_URL = os.getenv(
    "SCRAPER_DATABASE_URL", "postgresql://sulekha:sulekha@localhost:5432/sulekha"
)
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "sulekha-pdfs")
GCS_OUTPUT_BUCKET = os.getenv("GCS_OUTPUT_BUCKET", GCS_BUCKET)
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "parsed/")


@dataclass
class PDFRecord:
    pdf_id: str
    gcs_path: str
    gcs_bucket: str
    content_hash: str
    project_no: str
    project_name: str
    formulation: str
    expense: str
    lb_name: str
    district_name: str
    lb_type_label: str
    year_label: str


@dataclass
class ParseState:
    """Tracks processing state for checkpointing."""
    processed_ids: set
    failed_ids: dict  # pdf_id -> error message
    started_at: str
    last_updated: str
    total_processed: int
    total_failed: int
    
    @classmethod
    def load(cls, path: Path) -> "ParseState":
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return cls(
                processed_ids=set(data.get("processed_ids", [])),
                failed_ids=data.get("failed_ids", {}),
                started_at=data.get("started_at", datetime.now().isoformat()),
                last_updated=data.get("last_updated", datetime.now().isoformat()),
                total_processed=data.get("total_processed", 0),
                total_failed=data.get("total_failed", 0),
            )
        return cls(
            processed_ids=set(),
            failed_ids={},
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_processed=0,
            total_failed=0,
        )
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump({
                "processed_ids": list(self.processed_ids),
                "failed_ids": self.failed_ids,
                "started_at": self.started_at,
                "last_updated": self.last_updated,
                "total_processed": self.total_processed,
                "total_failed": self.total_failed,
            }, f, indent=2)
    
    def mark_processed(self, pdf_id: str):
        self.processed_ids.add(pdf_id)
        self.total_processed += 1
    
    def mark_failed(self, pdf_id: str, error: str):
        self.failed_ids[pdf_id] = error
        self.total_failed += 1
    
    def is_processed(self, pdf_id: str) -> bool:
        return pdf_id in self.processed_ids or pdf_id in self.failed_ids


def get_db_connection():
    return psycopg.connect(DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"))


def get_gcs_client():
    return storage.Client()


def query_pdfs(
    year_label: str,
    district: str | None = None,
    lb_type: str | None = None,
    limit: int | None = None,
) -> Iterator[PDFRecord]:
    """Query PDFs from scraper database."""
    sql = """
    SELECT 
        p.id as pdf_id,
        p.gcs_path,
        p.gcs_bucket,
        p.content_hash,
        pr.project_no,
        pr.project_name,
        pr.formulation,
        pr.expense,
        lb.lb_name,
        d.district_name,
        d.lb_type_label,
        d.year_label
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
      AND d.year_label = %s
    """
    params = [year_label]
    
    if district:
        sql += " AND d.district_name = %s"
        params.append(district)
    
    if lb_type:
        sql += " AND d.lb_type_label = %s"
        params.append(lb_type)
    
    sql += " ORDER BY d.district_name, lb.lb_name, pr.project_no"
    
    if limit:
        sql += f" LIMIT {limit}"
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            for row in cur:
                yield PDFRecord(
                    pdf_id=str(row[0]),
                    gcs_path=row[1],
                    gcs_bucket=row[2],
                    content_hash=row[3],
                    project_no=str(row[4]),
                    project_name=row[5],
                    formulation=str(row[6]) if row[6] else "",
                    expense=str(row[7]) if row[7] else "",
                    lb_name=row[8],
                    district_name=row[9],
                    lb_type_label=row[10],
                    year_label=row[11],
                )


def download_pdf(gcs_client, bucket_name: str, gcs_path: str, local_path: Path) -> bool:
    """Download PDF from GCS to local path."""
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))
        return True
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        return False


def upload_json(gcs_client, bucket_name: str, gcs_path: str, data: dict) -> bool:
    """Upload parsed JSON to GCS."""
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(
            json.dumps(data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        return True
    except Exception as e:
        console.print(f"[red]Upload failed: {e}[/red]")
        return False


def load_docling(layout_batch: int = 4, table_batch: int = 4):
    """Load Docling with GPU-optimized settings."""
    console.print("[yellow]Loading Docling with GPU optimization...[/yellow]")
    start = time.time()
    
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    
    # GPU batch size optimization
    # Higher values = faster but more VRAM
    # T4 (16GB): layout_batch=32-64, table_batch=4-8
    # A100 (40GB): layout_batch=64-128, table_batch=8-16
    if hasattr(pipeline_options, 'layout_batch_size'):
        pipeline_options.layout_batch_size = layout_batch
    if hasattr(pipeline_options, 'table_batch_size'):
        pipeline_options.table_batch_size = table_batch
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    elapsed = time.time() - start
    console.print(f"[green]Docling loaded in {elapsed:.1f}s[/green]")
    
    return converter


def parse_pdf(converter, pdf_path: Path, record: PDFRecord) -> dict:
    """Parse a single PDF with Docling."""
    start = time.time()
    
    result = converter.convert(pdf_path)
    doc = result.document
    
    elapsed = time.time() - start
    
    markdown = doc.export_to_markdown()
    
    blocks = []
    tables = []
    
    for item, level in doc.iterate_items():
        block_info = {
            "type": type(item).__name__,
            "level": level,
        }
        
        if hasattr(item, "text"):
            block_info["text"] = item.text[:1000] if item.text else ""
        
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, "page_no"):
                block_info["page"] = prov.page_no
        
        blocks.append(block_info)
        
        if "Table" in type(item).__name__:
            table_data = extract_table_data(item)
            if table_data:
                tables.append(table_data)
    
    page_count = doc.num_pages() if hasattr(doc, "num_pages") else 0
    
    return {
        "pdf_id": record.pdf_id,
        "filename": Path(record.gcs_path).name,
        "gcs_path": record.gcs_path,
        "content_hash": record.content_hash,
        "parse_time_seconds": elapsed,
        "page_count": page_count,
        "block_count": len(blocks),
        "table_count": len(tables),
        "metadata": {
            "project_no": record.project_no,
            "project_name": record.project_name,
            "formulation": record.formulation,
            "expense": record.expense,
            "lb_name": record.lb_name,
            "district_name": record.district_name,
            "lb_type_label": record.lb_type_label,
            "year_label": record.year_label,
        },
        "markdown": markdown,
        "blocks": blocks,
        "tables": tables,
        "parsed_at": datetime.now().isoformat(),
    }


def extract_table_data(table_item) -> dict | None:
    """Extract structured data from a Docling table."""
    try:
        if hasattr(table_item, "export_to_dataframe"):
            df = table_item.export_to_dataframe()
            headers = list(df.columns)
            rows = df.values.tolist()
            return {
                "headers": headers,
                "rows": rows[:50],  # Limit rows in output
                "total_rows": len(df),
            }
        return None
    except Exception:
        return None


def get_output_gcs_path(record: PDFRecord) -> str:
    """Generate GCS path for parsed output."""
    # parsed/2025-2026/Grama_Panchayat/Ernakulam/lb_name/project_no.json
    lb_name_safe = record.lb_name.replace(" ", "_").replace("/", "_")
    return f"{OUTPUT_PREFIX}{record.year_label}/{record.lb_type_label.replace(' ', '_')}/{record.district_name}/{lb_name_safe}/{record.project_no}.json"


def main():
    parser = argparse.ArgumentParser(description="GPU-optimized batch PDF parsing")
    parser.add_argument("--year", required=True, help="Year label (e.g., 2025-2026)")
    parser.add_argument("--district", help="Filter by district name")
    parser.add_argument("--lb-type", help="Filter by LB type")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to process")
    parser.add_argument("--layout-batch", type=int, default=32, help="Layout model batch size (default: 32)")
    parser.add_argument("--table-batch", type=int, default=4, help="Table model batch size (default: 4)")
    parser.add_argument("--state-file", default="parse_state.json", help="State file for checkpointing")
    parser.add_argument("--local-output", help="Save to local directory instead of GCS")
    parser.add_argument("--save-interval", type=int, default=10, help="Save state every N PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Count PDFs without processing")
    
    args = parser.parse_args()
    
    console.print("[bold green]Batch PDF Parser (GPU Optimized)[/bold green]\n")
    
    # Load state
    state_path = Path(args.state_file)
    state = ParseState.load(state_path)
    console.print(f"[dim]State file: {state_path}[/dim]")
    console.print(f"[dim]Previously processed: {len(state.processed_ids)}, Failed: {len(state.failed_ids)}[/dim]\n")
    
    # Query PDFs
    console.print(f"[yellow]Querying PDFs for {args.year}...[/yellow]")
    all_records = list(query_pdfs(
        year_label=args.year,
        district=args.district,
        lb_type=args.lb_type,
        limit=args.limit,
    ))
    
    pending = [r for r in all_records if not state.is_processed(r.pdf_id)]
    
    console.print(f"Total matching: {len(all_records)}")
    console.print(f"Already processed: {len(all_records) - len(pending)}")
    console.print(f"[bold]Pending: {len(pending)}[/bold]\n")
    
    if args.dry_run:
        console.print("[yellow]Dry run - exiting without processing[/yellow]")
        return
    
    if not pending:
        console.print("[green]All PDFs already processed![/green]")
        return
    
    # Initialize
    gcs_client = get_gcs_client()
    converter = load_docling(args.layout_batch, args.table_batch)
    
    local_output_dir = Path(args.local_output) if args.local_output else None
    if local_output_dir:
        local_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process PDFs
    console.print(f"\n[bold]Processing {len(pending)} PDFs...[/bold]\n")
    
    processed_count = 0
    failed_count = 0
    total_pages = 0
    total_time = 0.0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing PDFs", total=len(pending))
        
        for record in pending:
            progress.update(task, description=f"[cyan]{record.lb_name[:20]}[/cyan] #{record.project_no}")
            
            try:
                # Download PDF to temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                
                if not download_pdf(gcs_client, record.gcs_bucket, record.gcs_path, tmp_path):
                    state.mark_failed(record.pdf_id, "Download failed")
                    failed_count += 1
                    progress.advance(task)
                    continue
                
                # Parse
                parsed = parse_pdf(converter, tmp_path, record)
                
                # Upload or save locally
                if local_output_dir:
                    output_path = local_output_dir / f"{record.pdf_id}.json"
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)
                else:
                    output_gcs_path = get_output_gcs_path(record)
                    if not upload_json(gcs_client, GCS_OUTPUT_BUCKET, output_gcs_path, parsed):
                        state.mark_failed(record.pdf_id, "Upload failed")
                        failed_count += 1
                        progress.advance(task)
                        continue
                
                # Success
                state.mark_processed(record.pdf_id)
                processed_count += 1
                total_pages += parsed.get("page_count", 0)
                total_time += parsed.get("parse_time_seconds", 0)
                
                # Cleanup
                tmp_path.unlink(missing_ok=True)
                
            except Exception as e:
                state.mark_failed(record.pdf_id, str(e))
                failed_count += 1
                console.print(f"[red]Error: {record.pdf_id}: {e}[/red]")
            
            # Checkpoint
            if (processed_count + failed_count) % args.save_interval == 0:
                state.save(state_path)
            
            progress.advance(task)
    
    # Final save
    state.save(state_path)
    
    # Summary
    console.rule("[bold green]Summary")
    
    summary = Table(title="Batch Processing Results")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    
    summary.add_row("Processed this run", str(processed_count))
    summary.add_row("Failed this run", str(failed_count))
    summary.add_row("Total processed (all runs)", str(state.total_processed))
    summary.add_row("Total failed (all runs)", str(state.total_failed))
    summary.add_row("Total pages", str(total_pages))
    
    if total_time > 0:
        summary.add_row("Total parse time", f"{total_time:.1f}s")
        summary.add_row("Pages/second", f"{total_pages / total_time:.2f}")
        
        remaining = len(all_records) - state.total_processed - state.total_failed
        if remaining > 0:
            avg_time_per_pdf = total_time / max(processed_count, 1)
            eta_seconds = remaining * avg_time_per_pdf
            eta_hours = eta_seconds / 3600
            summary.add_row("Remaining PDFs", str(remaining))
            summary.add_row("ETA (at current rate)", f"{eta_hours:.1f} hours")
    
    console.print(summary)
    
    if local_output_dir:
        console.print(f"\n[bold green]Output saved to {local_output_dir}[/bold green]")
    else:
        console.print(f"\n[bold green]Output uploaded to gs://{GCS_OUTPUT_BUCKET}/{OUTPUT_PREFIX}[/bold green]")


if __name__ == "__main__":
    main()
