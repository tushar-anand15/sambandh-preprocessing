#!/usr/bin/env python3
"""
Benchmark comparison script for different machine configurations.

Run this after upgrading to 8-core machine to compare performance.
Uses the standard (non-threaded) pipeline which performed better on 4 cores.
"""

import os

# Limit Tesseract/OpenMP to 1 thread per process to prevent CPU contention.
# Note: Docling's ONNX/PyTorch models spawn ~1-2 additional threads per worker
# that cannot be suppressed post-fork. This is irreducible at ~1.6x load/cores.
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import psutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.table import Table

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
BENCHMARK_OUTPUT_DIR = Path(__file__).parent / "benchmark_results"


@dataclass
class BenchmarkResult:
    machine_config: str
    cpu_count: int
    ram_gb: float
    num_workers: int
    total_pdfs: int
    successful: int
    failed: int
    wall_time_seconds: float
    total_cpu_time_seconds: float
    speedup: float
    avg_time_per_pdf: float
    pdfs_per_hour: float
    timestamp: str


def get_system_info() -> dict:
    """Get current system configuration."""
    cpu_count = os.cpu_count() or 1
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":")[1].strip()
                    break
            else:
                cpu_model = "Unknown"
    except:
        cpu_model = "Unknown"
    
    return {
        "cpu_count": cpu_count,
        "cpu_model": cpu_model,
        "ram_gb": ram_gb,
    }


def get_converter():
    """Create Docling converter with standard (non-threaded) pipeline."""

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.datamodel.base_models import InputFormat

    ocr_options = TesseractCliOcrOptions(
        force_full_page_ocr=True,
        lang=["mal", "eng"],
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def process_single_pdf(pdf_path: Path) -> dict:
    """Process a single PDF."""
    worker_id = mp.current_process().name
    start_time = time.time()

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

        parse_time = time.time() - start_time
        print(f"[{worker_id}] Finished: {pdf_path.name} ({parse_time:.1f}s total)", flush=True)

        return {
            "filename": pdf_path.name,
            "status": "success",
            "parse_time_seconds": parse_time,
            "page_count": doc.num_pages(),
        }

    except Exception as e:
        parse_time = time.time() - start_time
        print(f"[{worker_id}] Error: {pdf_path.name} - {e} ({parse_time:.1f}s)", flush=True)
        return {
            "filename": pdf_path.name,
            "status": "error",
            "parse_time_seconds": parse_time,
            "error": str(e),
        }


def worker_process(pdf_path_str: str) -> dict:
    """Worker function."""
    return process_single_pdf(Path(pdf_path_str))


def monitor_contention(interval: float = 5.0, stop_event=None):
    """
    Background thread that logs CPU run queue depth and usage.
    Run queue length (r) > num_cores = contention.
    """
    import threading

    cpu_count = os.cpu_count() or 1
    samples = []

    while not (stop_event and stop_event.is_set()):
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        load_1, _, _ = os.getloadavg()
        # load avg > cpu_count = oversubscribed
        contention = load_1 / cpu_count
        samples.append({
            "ts": datetime.now().isoformat(),
            "load_1m": load_1,
            "contention_ratio": contention,
            "cpu_pct": cpu_pct,
            "mem_pct": vm.percent,
        })
        flag = " ⚠ CONTENDED" if contention > 1.5 else ""
        print(
            f"  [monitor] load={load_1:.1f} ({contention:.1f}x cores) "
            f"cpu={cpu_pct:.0f}% mem={vm.percent:.0f}%{flag}",
            flush=True,
        )
        time.sleep(interval)

    return samples


def run_benchmark(num_workers: int) -> BenchmarkResult:
    """Run benchmark with specified number of workers."""
    system_info = get_system_info()
    
    pdf_files = sorted(SAMPLE_PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDFs found in {SAMPLE_PDFS_DIR}")

    console.print(f"\n[bold]Running benchmark: {num_workers} workers[/bold]")
    console.print(f"  CPU: {system_info['cpu_model']}")
    console.print(f"  Cores: {system_info['cpu_count']}")
    console.print(f"  RAM: {system_info['ram_gb']:.1f} GB")
    console.print(f"  PDFs: {len(pdf_files)}")
    console.print()

    import threading

    batch_start = time.time()
    pdf_paths_str = [str(p) for p in pdf_files]

    # Start contention monitor in background thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_contention,
        args=(10.0, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    results = []
    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_process, pdf_paths_str)):
            results.append(result)
            status = "✓" if result["status"] == "success" else "✗"
            console.print(
                f"  {status} [{i+1}/{len(pdf_files)}] {result['filename']} ({result['parse_time_seconds']:.1f}s)"
            )

    stop_event.set()
    batch_end = time.time()
    wall_time = batch_end - batch_start

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    total_cpu_time = sum(r["parse_time_seconds"] for r in results)

    avg_time = total_cpu_time / len(results) if results else 0
    pdfs_per_hour = (3600 / avg_time) * num_workers * 0.90 if avg_time > 0 else 0

    machine_config = f"{system_info['cpu_count']}core_{int(system_info['ram_gb'])}gb"

    return BenchmarkResult(
        machine_config=machine_config,
        cpu_count=system_info["cpu_count"],
        ram_gb=system_info["ram_gb"],
        num_workers=num_workers,
        total_pdfs=len(results),
        successful=len(successful),
        failed=len(failed),
        wall_time_seconds=wall_time,
        total_cpu_time_seconds=total_cpu_time,
        speedup=total_cpu_time / wall_time if wall_time > 0 else 0,
        avg_time_per_pdf=avg_time,
        pdfs_per_hour=pdfs_per_hour,
        timestamp=datetime.now().isoformat(),
    )


def save_benchmark(result: BenchmarkResult):
    """Save benchmark result to file."""
    BENCHMARK_OUTPUT_DIR.mkdir(exist_ok=True)
    
    filename = f"benchmark_{result.machine_config}_{result.num_workers}w_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = BENCHMARK_OUTPUT_DIR / filename
    
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    console.print(f"\n[dim]Saved to {filepath}[/dim]")
    return filepath


def compare_benchmarks():
    """Compare all benchmark results."""
    if not BENCHMARK_OUTPUT_DIR.exists():
        console.print("[yellow]No benchmark results found[/yellow]")
        return

    results = []
    for f in BENCHMARK_OUTPUT_DIR.glob("benchmark_*.json"):
        with open(f) as fp:
            results.append(json.load(fp))

    if not results:
        console.print("[yellow]No benchmark results found[/yellow]")
        return

    results.sort(key=lambda x: x["timestamp"])

    table = Table(title="Benchmark Comparison")
    table.add_column("Config", style="cyan")
    table.add_column("Workers", justify="right")
    table.add_column("Wall Time", justify="right")
    table.add_column("Avg/PDF", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("PDFs/hr", justify="right", style="green")
    table.add_column("Timestamp", style="dim")

    for r in results:
        table.add_row(
            r["machine_config"],
            str(r["num_workers"]),
            f"{r['wall_time_seconds']/60:.1f}m",
            f"{r['avg_time_per_pdf']:.0f}s",
            f"{r['speedup']:.2f}x",
            f"{r['pdfs_per_hour']:.0f}",
            r["timestamp"][:16],
        )

    console.print(table)

    # Contention analysis
    console.print("\n[bold]Contention Analysis:[/bold]")
    cpu_count = os.cpu_count() or 1
    for r in results:
        ratio = r["num_workers"] / cpu_count
        status = "✓ OK" if ratio <= 1.0 else ("⚠ moderate" if ratio <= 2.0 else "✗ oversubscribed")
        console.print(f"  {r['machine_config']} {r['num_workers']}w / {r['cpu_count']} cores = {ratio:.1f}x  [{status}]")

    # Projection for Thrissur (9,294 PDFs)
    console.print("\n[bold]Projection for Thrissur District (9,294 PDFs):[/bold]")
    for r in results:
        hours = 9294 / r["pdfs_per_hour"] if r["pdfs_per_hour"] > 0 else float("inf")
        console.print(f"  {r['machine_config']} ({r['num_workers']}w): {hours:.1f} hours")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark PDF OCR processing")
    parser.add_argument("--workers", type=int, help="Number of workers (default: CPU_COUNT // 2)")
    parser.add_argument("--compare", action="store_true", help="Compare existing benchmarks")
    args = parser.parse_args()

    if args.compare:
        compare_benchmarks()
        return

    system_info = get_system_info()
    num_workers = args.workers or max(1, system_info["cpu_count"] // 2)

    console.print("[bold green]PDF OCR Benchmark[/bold green]")
    console.print(f"Machine: {system_info['cpu_count']} cores, {system_info['ram_gb']:.1f} GB RAM")

    result = run_benchmark(num_workers)
    save_benchmark(result)

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Wall time: {result.wall_time_seconds/60:.1f} minutes")
    console.print(f"  Avg per PDF: {result.avg_time_per_pdf:.0f}s")
    console.print(f"  Speedup: {result.speedup:.2f}x")
    console.print(f"  Throughput: {result.pdfs_per_hour:.0f} PDFs/hour")

    # Projection
    thrissur_pdfs = 9294
    hours_needed = thrissur_pdfs / result.pdfs_per_hour if result.pdfs_per_hour > 0 else float("inf")
    console.print(f"\n[bold]Thrissur projection ({thrissur_pdfs:,} PDFs):[/bold]")
    console.print(f"  Estimated time: {hours_needed:.1f} hours")


if __name__ == "__main__":
    main()
