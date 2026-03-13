#!/usr/bin/env python3
"""
01_db_counts.py — Explore scraper DB for PDF counts and distributions.

Queries the sulekha-scraper Postgres database to understand:
- Total PDFs per year
- Breakdown by lb_type × year
- Top districts by count for 2024-2025
- Sample rows with full metadata
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env")

console = Console()

DATABASE_URL = os.getenv(
    "SCRAPER_DATABASE_URL", "postgresql://sulekha:sulekha@localhost:5432/sulekha"
)


def get_connection():
    return psycopg.connect(DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"))


def query_pdfs_by_year(conn):
    """Total PDFs per year."""
    console.rule("[bold blue]PDFs by Year")
    
    sql = """
    SELECT 
        d.year_label,
        COUNT(DISTINCT p.id) as pdf_count,
        COUNT(DISTINCT pr.id) as project_count,
        COUNT(DISTINCT lb.id) as local_body_count
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
    GROUP BY d.year_label
    ORDER BY d.year_label DESC
    """
    
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    
    table = Table(title="PDFs by Year")
    table.add_column("Year", style="cyan")
    table.add_column("PDFs", justify="right", style="green")
    table.add_column("Projects", justify="right")
    table.add_column("Local Bodies", justify="right")
    
    for row in rows:
        table.add_row(row[0], f"{row[1]:,}", f"{row[2]:,}", f"{row[3]:,}")
    
    console.print(table)
    return rows


def query_by_lb_type_and_year(conn, year_label: str = "2024-2025"):
    """Breakdown by lb_type for a specific year."""
    console.rule(f"[bold blue]PDFs by LB Type for {year_label}")
    
    sql = """
    SELECT 
        d.lb_type_label,
        COUNT(DISTINCT p.id) as pdf_count
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
      AND d.year_label = %s
    GROUP BY d.lb_type_label
    ORDER BY pdf_count DESC
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (year_label,))
        rows = cur.fetchall()
    
    table = Table(title=f"PDFs by LB Type ({year_label})")
    table.add_column("LB Type", style="cyan")
    table.add_column("PDFs", justify="right", style="green")
    
    for row in rows:
        table.add_row(row[0], f"{row[1]:,}")
    
    console.print(table)
    return rows


def query_top_districts(conn, year_label: str = "2024-2025", limit: int = 5):
    """Top districts by PDF count for a specific year."""
    console.rule(f"[bold blue]Top {limit} Districts for {year_label}")
    
    sql = """
    SELECT 
        d.district_name,
        d.lb_type_label,
        COUNT(DISTINCT p.id) as pdf_count
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
      AND d.year_label = %s
    GROUP BY d.district_name, d.lb_type_label
    ORDER BY pdf_count DESC
    LIMIT %s
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (year_label, limit))
        rows = cur.fetchall()
    
    table = Table(title=f"Top {limit} Districts ({year_label})")
    table.add_column("District", style="cyan")
    table.add_column("LB Type", style="magenta")
    table.add_column("PDFs", justify="right", style="green")
    
    for row in rows:
        table.add_row(row[0], row[1], f"{row[2]:,}")
    
    console.print(table)
    return rows


def query_sample_rows(conn, year_label: str = "2024-2025", limit: int = 5):
    """Sample PDF rows with full metadata."""
    console.rule(f"[bold blue]Sample PDFs from {year_label}")
    
    sql = """
    SELECT 
        p.id as pdf_id,
        p.gcs_path,
        pr.project_no,
        pr.project_name,
        pr.formulation,
        pr.expense,
        lb.lb_name,
        d.district_name,
        d.lb_type_label
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
      AND d.year_label = %s
    ORDER BY RANDOM()
    LIMIT %s
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (year_label, limit))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    
    for i, row in enumerate(rows, 1):
        console.print(f"\n[bold yellow]Sample {i}:[/bold yellow]")
        for col, val in zip(columns, row):
            val_str = str(val)[:80] + "..." if len(str(val)) > 80 else str(val)
            console.print(f"  [cyan]{col}:[/cyan] {val_str}")
    
    return rows


def main():
    console.print("[bold green]Sambandh Preprocessing — DB Exploration[/bold green]\n")
    
    with get_connection() as conn:
        query_pdfs_by_year(conn)
        console.print()
        
        query_by_lb_type_and_year(conn, "2024-2025")
        console.print()
        
        query_by_lb_type_and_year(conn, "2025-2026")
        console.print()
        
        query_top_districts(conn, "2024-2025", limit=10)
        console.print()
        
        query_sample_rows(conn, "2024-2025", limit=5)
    
    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
