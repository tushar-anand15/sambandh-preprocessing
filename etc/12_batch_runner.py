#!/usr/bin/env python3
"""
Batch runner for partitioning PDFs across multiple runs or machines.

For testing with sample_pdfs (25 PDFs) or scaling to 80k PDFs.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

from rich.console import Console

console = Console()


def partition_files(
    pdf_dir: Path,
    num_partitions: int,
    partition_id: int,
) -> list[Path]:
    """
    Partition PDF files deterministically across N partitions.
    
    Uses hash-based partitioning so same file always goes to same partition.
    """
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    
    partition_files = []
    for pdf in all_pdfs:
        file_hash = int(hashlib.md5(pdf.name.encode()).hexdigest(), 16)
        if file_hash % num_partitions == partition_id:
            partition_files.append(pdf)
    
    return partition_files


def create_manifest(
    pdf_dir: Path,
    output_dir: Path,
    num_partitions: int = 1,
) -> dict:
    """Create a manifest file describing all partitions."""
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    
    partitions = {}
    for i in range(num_partitions):
        files = partition_files(pdf_dir, num_partitions, i)
        partitions[f"partition_{i}"] = {
            "id": i,
            "files": [f.name for f in files],
            "count": len(files),
        }
    
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(pdf_dir),
        "output_dir": str(output_dir),
        "total_files": len(all_pdfs),
        "num_partitions": num_partitions,
        "partitions": partitions,
    }
    
    manifest_path = output_dir / "_manifest.json"
    output_dir.mkdir(exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    console.print(f"[green]Manifest created: {manifest_path}[/green]")
    return manifest


def run_partition(
    pdf_dir: Path,
    output_dir: Path,
    num_partitions: int = 1,
    partition_id: int = 0,
    num_workers: int = 2,
):
    """Run processing for a specific partition."""
    from etc_11_parallel_threaded_ocr import main as run_ocr
    
    files = partition_files(pdf_dir, num_partitions, partition_id)
    
    console.print(f"[bold]Partition {partition_id}/{num_partitions}[/bold]")
    console.print(f"  Files: {len(files)}")
    console.print(f"  Workers: {num_workers}")
    
    partition_output = output_dir / f"partition_{partition_id}"
    partition_output.mkdir(exist_ok=True)
    
    for pdf in files:
        (partition_output / pdf.name).symlink_to(pdf)
    
    run_ocr(
        input_dir=partition_output,
        output_dir=partition_output,
        num_workers=num_workers,
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch runner for PDF OCR")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    manifest_parser = subparsers.add_parser("manifest", help="Create partition manifest")
    manifest_parser.add_argument("--input-dir", type=Path, required=True)
    manifest_parser.add_argument("--output-dir", type=Path, required=True)
    manifest_parser.add_argument("--partitions", type=int, default=1)
    
    run_parser = subparsers.add_parser("run", help="Run a specific partition")
    run_parser.add_argument("--input-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--partitions", type=int, default=1)
    run_parser.add_argument("--partition-id", type=int, default=0)
    run_parser.add_argument("--workers", type=int, default=2)
    
    info_parser = subparsers.add_parser("info", help="Show partition info")
    info_parser.add_argument("--input-dir", type=Path, required=True)
    info_parser.add_argument("--partitions", type=int, default=1)
    
    args = parser.parse_args()
    
    if args.command == "manifest":
        create_manifest(args.input_dir, args.output_dir, args.partitions)
    
    elif args.command == "run":
        run_partition(
            args.input_dir,
            args.output_dir,
            args.partitions,
            args.partition_id,
            args.workers,
        )
    
    elif args.command == "info":
        all_pdfs = sorted(args.input_dir.glob("*.pdf"))
        console.print(f"[bold]Total PDFs: {len(all_pdfs)}[/bold]\n")
        
        for i in range(args.partitions):
            files = partition_files(args.input_dir, args.partitions, i)
            console.print(f"Partition {i}: {len(files)} files")
            for f in files[:5]:
                console.print(f"  - {f.name}")
            if len(files) > 5:
                console.print(f"  ... and {len(files) - 5} more")


if __name__ == "__main__":
    main()
