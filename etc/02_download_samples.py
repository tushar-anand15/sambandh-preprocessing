#!/usr/bin/env python3
"""
02_download_samples.py — Collect diverse sample PDFs for testing.

Selects PDFs from 2025-2026 covering:
- All 14 districts
- All 5 LB types (5 PDFs each = 25 total minimum)

First checks local test_pdfs/ directory, falls back to GCS download.
"""

import os
import shutil
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from google.cloud import storage
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env")

console = Console()

DATABASE_URL = os.getenv(
    "SCRAPER_DATABASE_URL", "postgresql://sulekha:sulekha@localhost:5432/sulekha"
)
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "sulekha-pdfs")
TEST_PDFS_PATH = Path(os.getenv("TEST_PDFS_PATH", "/home/tusharanand/sulekha-scraper/test_pdfs/pdfs"))
SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"

YEAR_LABEL = "2025-2026"
SAMPLES_PER_LB_TYPE = 5

LB_TYPES = [
    "Grama Panchayat",
    "Municipality",
    "Block Panchayat",
    "District Panchayat",
    "Corporation",
]


def get_connection():
    return psycopg.connect(DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"))


def get_gcs_client():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    return storage.Client()


def query_sample_pdfs_by_lb_type(conn, lb_type: str, limit: int = 5):
    """Get sample PDFs for a specific LB type, trying to cover different districts."""
    sql = """
    WITH ranked AS (
        SELECT 
            p.id as pdf_id,
            p.gcs_path,
            p.gcs_bucket,
            p.content_hash,
            pr.project_no,
            pr.project_name,
            lb.lb_name,
            d.district_name,
            d.lb_type_label,
            ROW_NUMBER() OVER (PARTITION BY d.district_name ORDER BY RANDOM()) as rn
        FROM pdfs p
        JOIN projects pr ON pr.id = p.project_id
        JOIN local_bodies lb ON lb.id = pr.local_body_id
        JOIN districts d ON d.id = lb.district_id
        WHERE p.status = 'UPLOADED'
          AND d.year_label = %s
          AND d.lb_type_label = %s
    )
    SELECT pdf_id, gcs_path, gcs_bucket, content_hash, project_no, 
           project_name, lb_name, district_name, lb_type_label
    FROM ranked
    WHERE rn = 1
    ORDER BY district_name
    LIMIT %s
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (YEAR_LABEL, lb_type, limit))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    
    return [dict(zip(columns, row)) for row in rows]


def find_local_pdf(gcs_path: str) -> Path | None:
    """Check if PDF exists in local test_pdfs directory."""
    relative_path = gcs_path.replace("pdfs/", "", 1) if gcs_path.startswith("pdfs/") else gcs_path
    local_path = TEST_PDFS_PATH / relative_path
    
    if local_path.exists():
        return local_path
    return None


def download_from_gcs(gcs_bucket: str, gcs_path: str, dest_path: Path) -> bool:
    """Download PDF from GCS."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest_path))
        return True
    except Exception as e:
        console.print(f"[red]GCS download failed: {e}[/red]")
        return False


def generate_sample_filename(pdf_info: dict) -> str:
    """Generate a descriptive filename for the sample PDF."""
    lb_type_short = pdf_info["lb_type_label"].replace(" ", "_")[:10]
    district_short = pdf_info["district_name"][:8]
    project_no = pdf_info["project_no"]
    return f"{lb_type_short}_{district_short}_{project_no}.pdf"


def collect_samples():
    """Main function to collect sample PDFs."""
    console.print(f"[bold green]Collecting Sample PDFs for {YEAR_LABEL}[/bold green]\n")
    
    SAMPLE_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    with get_connection() as conn:
        for lb_type in LB_TYPES:
            console.rule(f"[bold blue]{lb_type}")
            
            samples = query_sample_pdfs_by_lb_type(conn, lb_type, SAMPLES_PER_LB_TYPE)
            console.print(f"Found {len(samples)} samples from DB")
            
            for sample in samples:
                sample_filename = generate_sample_filename(sample)
                dest_path = SAMPLE_PDFS_DIR / sample_filename
                
                local_path = find_local_pdf(sample["gcs_path"])
                
                if local_path:
                    shutil.copy(local_path, dest_path)
                    sample["source"] = "local"
                    sample["local_path"] = str(dest_path)
                    console.print(f"  [green]✓[/green] {sample_filename} (local copy)")
                else:
                    success = download_from_gcs(
                        sample["gcs_bucket"], 
                        sample["gcs_path"], 
                        dest_path
                    )
                    if success:
                        sample["source"] = "gcs"
                        sample["local_path"] = str(dest_path)
                        console.print(f"  [green]✓[/green] {sample_filename} (GCS download)")
                    else:
                        sample["source"] = "failed"
                        sample["local_path"] = None
                        console.print(f"  [red]✗[/red] {sample_filename} (failed)")
                
                all_samples.append(sample)
            
            console.print()
    
    return all_samples


def print_summary(samples: list[dict]):
    """Print summary of collected samples."""
    console.rule("[bold green]Summary")
    
    table = Table(title=f"Collected Samples ({YEAR_LABEL})")
    table.add_column("LB Type", style="cyan")
    table.add_column("District", style="magenta")
    table.add_column("Project", style="white")
    table.add_column("Source", style="yellow")
    table.add_column("File", style="green")
    
    for s in samples:
        if s.get("local_path"):
            filename = Path(s["local_path"]).name
            table.add_row(
                s["lb_type_label"][:15],
                s["district_name"][:12],
                str(s["project_no"]),
                s["source"],
                filename[:30]
            )
    
    console.print(table)
    
    by_lb_type = {}
    for s in samples:
        lb = s["lb_type_label"]
        by_lb_type[lb] = by_lb_type.get(lb, 0) + (1 if s.get("local_path") else 0)
    
    console.print("\n[bold]By LB Type:[/bold]")
    for lb, count in by_lb_type.items():
        console.print(f"  {lb}: {count} PDFs")
    
    by_source = {"local": 0, "gcs": 0, "failed": 0}
    for s in samples:
        by_source[s.get("source", "failed")] += 1
    
    console.print("\n[bold]By Source:[/bold]")
    console.print(f"  Local: {by_source['local']}")
    console.print(f"  GCS: {by_source['gcs']}")
    console.print(f"  Failed: {by_source['failed']}")
    
    total_success = by_source["local"] + by_source["gcs"]
    console.print(f"\n[bold green]Total: {total_success} PDFs collected in {SAMPLE_PDFS_DIR}[/bold green]")


def main():
    samples = collect_samples()
    print_summary(samples)


if __name__ == "__main__":
    main()
