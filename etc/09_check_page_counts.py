#!/usr/bin/env python3
"""
Check if the scraper DB has page counts for PDFs.
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from rich.console import Console

load_dotenv(Path(__file__).parent.parent / ".env")

console = Console()

DATABASE_URL = os.getenv(
    "SCRAPER_DATABASE_URL", "postgresql://sulekha:sulekha@localhost:5432/sulekha"
)


def get_connection():
    return psycopg.connect(DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"))


def check_pdf_schema(conn):
    """Check the pdfs table schema."""
    console.rule("[bold blue]PDFs Table Schema")
    
    sql = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'pdfs'
    ORDER BY ordinal_position
    """
    
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    
    for row in rows:
        console.print(f"  {row[0]}: {row[1]} (nullable: {row[2]})")
    
    return rows


def check_projects_schema(conn):
    """Check the projects table schema."""
    console.rule("[bold blue]Projects Table Schema")
    
    sql = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'projects'
    ORDER BY ordinal_position
    """
    
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    
    for row in rows:
        console.print(f"  {row[0]}: {row[1]} (nullable: {row[2]})")
    
    return rows


def query_2025_2026_stats(conn):
    """Get stats for 2025-2026 year."""
    console.rule("[bold blue]2025-2026 Year Stats")
    
    sql = """
    SELECT 
        COUNT(DISTINCT p.id) as pdf_count,
        COUNT(DISTINCT pr.id) as project_count,
        COUNT(DISTINCT lb.id) as local_body_count,
        COUNT(DISTINCT d.district_name) as district_count
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
      AND d.year_label = '2025-2026'
    """
    
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
    
    console.print(f"  PDFs: {row[0]:,}")
    console.print(f"  Projects: {row[1]:,}")
    console.print(f"  Local Bodies: {row[2]:,}")
    console.print(f"  Districts: {row[3]:,}")
    
    return row


def query_by_lb_type(conn):
    """Breakdown by LB type for 2025-2026."""
    console.rule("[bold blue]2025-2026 by LB Type")
    
    sql = """
    SELECT 
        d.lb_type_label,
        COUNT(DISTINCT p.id) as pdf_count
    FROM pdfs p
    JOIN projects pr ON pr.id = p.project_id
    JOIN local_bodies lb ON lb.id = pr.local_body_id
    JOIN districts d ON d.id = lb.district_id
    WHERE p.status = 'UPLOADED'
      AND d.year_label = '2025-2026'
    GROUP BY d.lb_type_label
    ORDER BY pdf_count DESC
    """
    
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    
    total = 0
    for row in rows:
        console.print(f"  {row[0]}: {row[1]:,}")
        total += row[1]
    
    console.print(f"  [bold]Total: {total:,}[/bold]")
    
    return rows


def sample_pdf_with_all_fields(conn):
    """Get a sample PDF row with all fields."""
    console.rule("[bold blue]Sample PDF Row (all fields)")
    
    sql = """
    SELECT *
    FROM pdfs
    WHERE status = 'UPLOADED'
    LIMIT 1
    """
    
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
    
    for col, val in zip(columns, row):
        val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
        console.print(f"  {col}: {val_str}")
    
    return dict(zip(columns, row))


def main():
    console.print("[bold green]Checking DB Schema and Page Counts[/bold green]\n")
    
    with get_connection() as conn:
        check_pdf_schema(conn)
        console.print()
        
        check_projects_schema(conn)
        console.print()
        
        query_2025_2026_stats(conn)
        console.print()
        
        query_by_lb_type(conn)
        console.print()
        
        sample_pdf_with_all_fields(conn)
    
    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
