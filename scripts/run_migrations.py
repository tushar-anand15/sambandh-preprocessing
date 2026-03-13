#!/usr/bin/env python3
"""
Simple migration runner for sambandh database.

Usage:
    python scripts/run_migrations.py [--dry-run] [--bootstrap]

Migrations are stored in scripts/migrations/ as numbered SQL files.
Tracks applied migrations in a schema_migrations table.

For existing databases that were created before the migration system,
use --bootstrap to mark the initial migration as applied without running it.
"""

import argparse
import os
from pathlib import Path

import psycopg
from rich.console import Console

console = Console()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://sambandh:sambandh@localhost:5433/sambandh"
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    row = conn.execute(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = %s
        ) AS exists
        """,
        [table_name],
    ).fetchone()
    return row["exists"]


def get_applied_migrations(conn) -> set[str]:
    """Get list of already-applied migration names."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    rows = conn.execute("SELECT name FROM schema_migrations").fetchall()
    return {row["name"] for row in rows}


def mark_migration_applied(conn, name: str):
    """Mark a migration as applied without running it."""
    conn.execute(
        "INSERT INTO schema_migrations (name) VALUES (%s) ON CONFLICT DO NOTHING",
        [name],
    )


def apply_migration(conn, migration_path: Path, dry_run: bool = False):
    """Apply a single migration file."""
    name = migration_path.stem
    sql = migration_path.read_text()

    if dry_run:
        console.print(f"[yellow]Would apply:[/yellow] {name}")
        console.print(f"[dim]{sql[:200]}...[/dim]" if len(sql) > 200 else f"[dim]{sql}[/dim]")
        return

    console.print(f"[cyan]Applying:[/cyan] {name}")
    conn.execute(sql)
    conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", [name])
    console.print(f"[green]Applied:[/green] {name}")


def bootstrap_existing_db(conn, migration_files: list[Path], dry_run: bool = False):
    """
    Bootstrap an existing database that was created before migrations.
    Marks 000_initial_schema as applied if parse_jobs table exists.
    """
    if not table_exists(conn, "parse_jobs"):
        console.print("[yellow]No existing tables found. Running all migrations.[/yellow]")
        return False

    initial_migration = next((f for f in migration_files if f.stem == "000_initial_schema"), None)
    if not initial_migration:
        console.print("[yellow]No 000_initial_schema migration found.[/yellow]")
        return False

    applied = get_applied_migrations(conn)
    if "000_initial_schema" in applied:
        console.print("[green]Database already bootstrapped.[/green]")
        return True

    if dry_run:
        console.print("[yellow]Would mark 000_initial_schema as applied (existing DB)[/yellow]")
    else:
        console.print("[cyan]Bootstrapping:[/cyan] Marking 000_initial_schema as applied (existing DB)")
        mark_migration_applied(conn, "000_initial_schema")
        conn.commit()
        console.print("[green]Bootstrapped:[/green] 000_initial_schema marked as applied")

    return True


def main():
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be applied")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap existing DB by marking initial migration as applied",
    )
    args = parser.parse_args()

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not migration_files:
        console.print("[yellow]No migration files found in scripts/migrations/[/yellow]")
        return

    console.print(f"[bold]Found {len(migration_files)} migration file(s)[/bold]\n")

    with psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.dict_row) as conn:
        if args.bootstrap:
            bootstrap_existing_db(conn, migration_files, args.dry_run)

        applied = get_applied_migrations(conn)

        pending = [f for f in migration_files if f.stem not in applied]

        if not pending:
            console.print("[green]All migrations already applied.[/green]")
            return

        console.print(f"[bold]{len(pending)} pending migration(s):[/bold]")
        for f in pending:
            console.print(f"  - {f.stem}")
        console.print()

        for migration_path in pending:
            apply_migration(conn, migration_path, args.dry_run)

        if not args.dry_run:
            conn.commit()
            console.print(
                f"\n[bold green]Successfully applied {len(pending)} migration(s)[/bold green]"
            )


if __name__ == "__main__":
    main()
