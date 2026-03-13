#!/usr/bin/env python3
"""
Parallel Tesseract OCR extraction using Docling's ThreadedPdfPipelineOptions.

Combines:
- Process-level parallelism (2 workers for 16GB RAM)
- Docling's threaded pipeline for intra-document parallelism (~40% speedup)
- Checkpointing for resume capability
- Memory monitoring
"""

import os

os.environ["OMP_THREAD_LIMIT"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
import time
import psutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# Configuration
CPU_COUNT = os.cpu_count() or 4
# With ThreadedPdfPipeline, each worker uses multiple threads internally
# Rule: max workers = CPU_COUNT // 2 (minimum 1)
NUM_WORKERS = max(1, CPU_COUNT // 2)
OCR_BATCH_SIZE = 4
LAYOUT_BATCH_SIZE = 4
TABLE_BATCH_SIZE = 4
QUEUE_MAX_SIZE = 50
CHECKPOINT_INTERVAL = 10
MEMORY_THRESHOLD_PERCENT = 85


@dataclass
class ProcessingResult:
    filename: str
    status: str
    parse_time_seconds: float
    page_count: Optional[int] = None
    block_count: Optional[int] = None
    table_count: Optional[int] = None
    error: Optional[str] = None
    memory_mb: Optional[float] = None


@dataclass
class Checkpoint:
    started_at: str
    last_updated: str
    total_files: int
    completed: list
    failed: list
    in_progress: list


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_system_memory_percent() -> float:
    """Get system memory usage percentage."""
    return psutil.virtual_memory().percent


def load_checkpoint(checkpoint_path: Path) -> Optional[Checkpoint]:
    """Load checkpoint from file if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
            return Checkpoint(**data)
    return None


def save_checkpoint(checkpoint_path: Path, checkpoint: Checkpoint):
    """Save checkpoint to file."""
    checkpoint.last_updated = datetime.now().isoformat()
    with open(checkpoint_path, "w") as f:
        json.dump(asdict(checkpoint), f, indent=2)


def get_converter(use_threaded: bool = True):
    """Create Docling converter with optional threading."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.datamodel.base_models import InputFormat

    ocr_options = TesseractCliOcrOptions(
        force_full_page_ocr=True,
        lang=["mal", "eng"],
    )

    if use_threaded:
        from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions

        pipeline_options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                num_threads=2,
                device="cpu",
            ),
            ocr_batch_size=OCR_BATCH_SIZE,
            layout_batch_size=LAYOUT_BATCH_SIZE,
            table_batch_size=TABLE_BATCH_SIZE,
            queue_max_size=QUEUE_MAX_SIZE,
        )
    else:
        pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


# Global flag for threaded mode (set by CLI)
USE_THREADED_PIPELINE = True


def process_single_pdf(pdf_path: Path, output_dir: Path) -> ProcessingResult:
    """Process a single PDF with Docling's threaded pipeline."""
    worker_id = mp.current_process().name
    start_time = time.time()
    start_memory = get_memory_usage_mb()

    print(f"[{worker_id}] Starting: {pdf_path.name}", flush=True)

    try:
        converter = get_converter()
        load_time = time.time() - start_time
        print(f"[{worker_id}] {pdf_path.name}: Docling loaded ({load_time:.1f}s)", flush=True)

        convert_start = time.time()
        result = converter.convert(str(pdf_path))
        doc = result.document

        convert_time = time.time() - convert_start
        print(
            f"[{worker_id}] {pdf_path.name}: Converted {doc.num_pages()} pages ({convert_time:.1f}s)",
            flush=True,
        )

        markdown = doc.export_to_markdown()

        blocks = []
        for item, level in doc.iterate_items():
            block_info = {
                "type": type(item).__name__,
                "level": level,
            }
            if hasattr(item, "text"):
                block_info["text"] = item.text
            if hasattr(item, "prov") and item.prov:
                block_info["page"] = item.prov[0].page_no if item.prov else None
            blocks.append(block_info)

        tables = []
        for table in doc.tables:
            table_data = table.export_to_dataframe(doc)
            tables.append(
                {
                    "headers": list(table_data.columns),
                    "rows": table_data.values.tolist(),
                    "total_rows": len(table_data),
                }
            )

        parse_time = time.time() - start_time
        end_memory = get_memory_usage_mb()

        output_data = {
            "filename": pdf_path.name,
            "status": "success",
            "extractor": "docling_threaded_ocr",
            "ocr_engine": "tesseract",
            "ocr_languages": ["mal", "eng"],
            "parse_time_seconds": parse_time,
            "page_count": doc.num_pages(),
            "block_count": len(blocks),
            "table_count": len(tables),
            "markdown": markdown,
            "blocks": blocks,
            "tables": tables,
        }

        output_path = output_dir / f"{pdf_path.stem}.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[{worker_id}] Finished: {pdf_path.name} ({parse_time:.1f}s total)", flush=True)

        return ProcessingResult(
            filename=pdf_path.name,
            status="success",
            parse_time_seconds=parse_time,
            page_count=doc.num_pages(),
            block_count=len(blocks),
            table_count=len(tables),
            memory_mb=end_memory - start_memory,
        )

    except Exception as e:
        parse_time = time.time() - start_time
        print(f"[{worker_id}] Error: {pdf_path.name} - {e} ({parse_time:.1f}s)", flush=True)

        return ProcessingResult(
            filename=pdf_path.name,
            status="error",
            parse_time_seconds=parse_time,
            error=str(e),
        )


def worker_process(args: tuple) -> ProcessingResult:
    """Worker function that processes a PDF and returns result."""
    pdf_path_str, output_dir_str = args
    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)
    return process_single_pdf(pdf_path, output_dir)


def check_memory_pressure() -> bool:
    """Check if system is under memory pressure."""
    mem_percent = get_system_memory_percent()
    if mem_percent > MEMORY_THRESHOLD_PERCENT:
        console.print(
            f"[yellow]Warning: Memory usage at {mem_percent:.1f}% (threshold: {MEMORY_THRESHOLD_PERCENT}%)[/yellow]"
        )
        return True
    return False


class AdaptiveWorkerPool:
    """Pool that adapts worker count based on memory pressure."""

    def __init__(self, max_workers: int, min_workers: int = 1):
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.current_workers = max_workers
        self.memory_samples = []
        self.sample_window = 5

    def get_recommended_workers(self) -> int:
        """Get recommended worker count based on memory usage."""
        mem_percent = get_system_memory_percent()
        self.memory_samples.append(mem_percent)
        if len(self.memory_samples) > self.sample_window:
            self.memory_samples.pop(0)

        avg_memory = sum(self.memory_samples) / len(self.memory_samples)

        if avg_memory > 90:
            recommended = self.min_workers
        elif avg_memory > 80:
            recommended = max(self.min_workers, self.max_workers - 1)
        elif avg_memory > 70:
            recommended = max(self.min_workers, self.max_workers)
        else:
            recommended = self.max_workers

        if recommended != self.current_workers:
            console.print(
                f"[cyan]Memory at {avg_memory:.1f}%: adjusting workers {self.current_workers} -> {recommended}[/cyan]"
            )
            self.current_workers = recommended

        return self.current_workers

    def should_pause(self) -> bool:
        """Check if processing should pause due to critical memory."""
        mem_percent = get_system_memory_percent()
        return mem_percent > 95

    def wait_for_memory(self, timeout: int = 60):
        """Wait for memory to recover."""
        start = time.time()
        while self.should_pause() and (time.time() - start) < timeout:
            console.print(
                f"[red]Critical memory ({get_system_memory_percent():.1f}%), waiting...[/red]"
            )
            time.sleep(5)


def get_memory_stats() -> dict:
    """Get detailed memory statistics."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": vm.total / (1024**3),
        "available_gb": vm.available / (1024**3),
        "used_gb": vm.used / (1024**3),
        "percent": vm.percent,
    }


def main(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    num_workers: int = NUM_WORKERS,
    adaptive: bool = True,
):
    """Main processing function with checkpointing and memory monitoring."""
    if input_dir is None:
        input_dir = Path(__file__).parent / "sample_pdfs"
    if output_dir is None:
        output_dir = Path(__file__).parent / "sample_output_threaded"

    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "_checkpoint.json"

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[red]No PDFs found in {input_dir}[/red]")
        return

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        checkpoint = Checkpoint(
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_files=len(pdf_files),
            completed=[],
            failed=[],
            in_progress=[],
        )

    completed_set = set(checkpoint.completed)
    to_process = [p for p in pdf_files if p.name not in completed_set]

    if not to_process:
        console.print(f"[green]All {len(pdf_files)} PDFs already processed![/green]")
        return

    mem_stats = get_memory_stats()
    console.print(f"[bold]Processing {len(to_process)} PDFs with {num_workers} workers[/bold]")
    console.print(f"[dim]Using ThreadedPdfPipelineOptions (batch_size={OCR_BATCH_SIZE})[/dim]")
    console.print(f"[dim]Skipping {len(completed_set)} already completed[/dim]")
    console.print(
        f"[dim]System memory: {mem_stats['used_gb']:.1f}GB / {mem_stats['total_gb']:.1f}GB "
        f"({mem_stats['percent']:.1f}% used)[/dim]"
    )
    console.print(f"[dim]Adaptive worker scaling: {'enabled' if adaptive else 'disabled'}[/dim]\n")

    adaptive_pool = AdaptiveWorkerPool(max_workers=num_workers, min_workers=1)
    batch_start = time.time()
    results = []
    processed_since_checkpoint = 0

    args_list = [(str(p), str(output_dir)) for p in to_process]

    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_process, args_list)):
            results.append(result)

            if result.status == "success":
                checkpoint.completed.append(result.filename)
                status_icon = "[green]✓[/green]"
                details = (
                    f"{result.page_count} pages, {result.block_count} blocks, "
                    f"{result.table_count} tables"
                )
            else:
                checkpoint.failed.append(result.filename)
                status_icon = "[red]✗[/red]"
                details = f"Error - {result.error[:50]}..." if result.error else "Unknown error"

            mem_stats = get_memory_stats()
            console.print(
                f"{status_icon} [{i+1}/{len(to_process)}] {result.filename}: "
                f"{details} ({result.parse_time_seconds:.1f}s) "
                f"[dim]mem: {mem_stats['percent']:.0f}%[/dim]"
            )

            processed_since_checkpoint += 1
            if processed_since_checkpoint >= CHECKPOINT_INTERVAL:
                save_checkpoint(checkpoint_path, checkpoint)
                processed_since_checkpoint = 0
                console.print(f"[dim]Checkpoint saved ({len(checkpoint.completed)} completed)[/dim]")

            if adaptive:
                if adaptive_pool.should_pause():
                    adaptive_pool.wait_for_memory()
                adaptive_pool.get_recommended_workers()

    save_checkpoint(checkpoint_path, checkpoint)

    batch_end = time.time()
    total_time = batch_end - batch_start

    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]
    total_parse_time = sum(r.parse_time_seconds for r in results)

    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print("[bold green]Batch Complete[/bold green]")
    console.print(f"[bold]{'='*60}[/bold]")
    console.print(f"  Workers:        {num_workers}")
    console.print(f"  PDFs processed: {len(results)}")
    console.print(f"  Successful:     {len(successful)}")
    console.print(f"  Failed:         {len(failed)}")
    console.print(f"  Wall time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    console.print(f"  Total CPU time: {total_parse_time:.1f}s ({total_parse_time/60:.1f} min)")
    console.print(f"  Speedup:        {total_parse_time/total_time:.2f}x")

    if successful:
        avg_time = sum(r.parse_time_seconds for r in successful) / len(successful)
        total_pages = sum(r.page_count or 0 for r in successful)
        console.print(f"  Avg per PDF:    {avg_time:.1f}s")
        console.print(f"  Total pages:    {total_pages}")
        console.print(f"  Avg per page:   {total_time/total_pages:.1f}s")

    summary = {
        "batch_start": checkpoint.started_at,
        "batch_end": datetime.now().isoformat(),
        "num_workers": num_workers,
        "config": {
            "ocr_batch_size": OCR_BATCH_SIZE,
            "layout_batch_size": LAYOUT_BATCH_SIZE,
            "table_batch_size": TABLE_BATCH_SIZE,
            "queue_max_size": QUEUE_MAX_SIZE,
        },
        "total_pdfs": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "wall_time_seconds": total_time,
        "total_cpu_time_seconds": total_parse_time,
        "speedup": total_parse_time / total_time if total_time > 0 else 0,
        "results": [asdict(r) for r in results],
    }

    summary_path = output_dir / "_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n[dim]Summary saved to {summary_path}[/dim]")
    console.print(f"[dim]Checkpoint saved to {checkpoint_path}[/dim]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel PDF OCR with Docling")
    parser.add_argument("--input-dir", type=Path, help="Input directory with PDFs")
    parser.add_argument("--output-dir", type=Path, help="Output directory for JSON results")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable adaptive worker scaling based on memory",
    )
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.workers,
        adaptive=not args.no_adaptive,
    )
