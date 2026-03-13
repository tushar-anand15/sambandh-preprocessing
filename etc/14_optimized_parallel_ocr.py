#!/usr/bin/env python3
"""
Optimized parallel Tesseract OCR extraction for Malayalam PDFs using Docling.

Optimizations over 10_parallel_tesseract_test.py:
  1. Full thread suppression (OMP + ORT + TORCH) to reduce contention
  2. Pre-import Docling in parent so fork inherits loaded modules
  3. Pool initializer: DocumentConverter created once per worker, not per PDF
  4. PDFs sorted by file size descending to minimize tail latency
"""

import os

# --- Thread suppression: must be set before any numeric library imports ---
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["ORT_NUM_INTER_THREADS"] = "1"
os.environ["ORT_NUM_INTRA_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import json
import time
import threading
import multiprocessing as mp
import psutil
from pathlib import Path
from datetime import datetime

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Pre-import Docling in parent so forked children inherit the loaded modules.
# This avoids ~9s model load on each worker's first task.
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat

from rich.console import Console

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
SAMPLE_OUTPUT_DIR = Path(__file__).parent / "sample_output_optimized"


def _make_converter() -> DocumentConverter:
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


# --- Pool initializer: converter loaded once per worker process ---
_converter: DocumentConverter | None = None


def _init_worker():
    global _converter
    _converter = _make_converter()
    print(f"[{mp.current_process().name}] Converter initialized", flush=True)


def process_single_pdf(pdf_path_str: str) -> dict:
    pdf_path = Path(pdf_path_str)
    worker_id = mp.current_process().name
    start_time = time.time()
    print(f"[{worker_id}] Starting: {pdf_path.name}", flush=True)

    try:
        result = _converter.convert(str(pdf_path))
        doc = result.document

        convert_time = time.time() - start_time
        print(
            f"[{worker_id}] {pdf_path.name}: Converted {doc.num_pages()} pages ({convert_time:.1f}s)",
            flush=True,
        )

        markdown = doc.export_to_markdown()

        blocks = []
        for item, level in doc.iterate_items():
            block_info = {"type": type(item).__name__, "level": level}
            if hasattr(item, "text"):
                block_info["text"] = item.text
            if hasattr(item, "prov") and item.prov:
                block_info["page"] = item.prov[0].page_no if item.prov else None
            blocks.append(block_info)

        tables = []
        for table in doc.tables:
            table_data = table.export_to_dataframe(doc)
            tables.append({
                "headers": list(table_data.columns),
                "rows": table_data.values.tolist(),
                "total_rows": len(table_data),
            })

        parse_time = time.time() - start_time
        print(f"[{worker_id}] Finished: {pdf_path.name} ({parse_time:.1f}s total)", flush=True)

        return {
            "filename": pdf_path.name,
            "status": "success",
            "extractor": "docling_ocr",
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
    except Exception as e:
        parse_time = time.time() - start_time
        print(f"[{worker_id}] Error: {pdf_path.name} - {e} ({parse_time:.1f}s)", flush=True)
        return {
            "filename": pdf_path.name,
            "status": "error",
            "error": str(e),
            "parse_time_seconds": parse_time,
        }


def monitor_system(interval: float = 10.0, stop_event: threading.Event = None):
    cpu_count = os.cpu_count() or 1
    while not (stop_event and stop_event.is_set()):
        load_1, _, _ = os.getloadavg()
        ratio = load_1 / cpu_count
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        flag = " ⚠ CONTENDED" if ratio > 1.5 else ""
        print(
            f"  [monitor] load={load_1:.1f} ({ratio:.1f}x cores) "
            f"cpu={cpu_pct:.0f}% mem={vm.percent:.0f}%{flag}",
            flush=True,
        )
        time.sleep(interval)


def main(
    input_dir: Path = SAMPLE_PDFS_DIR,
    output_dir: Path = SAMPLE_OUTPUT_DIR,
    num_workers: int = 8,
    reprocess: bool = False,
):
    output_dir.mkdir(exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDFs found in {input_dir}[/red]")
        return

    to_process = []
    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.json"
        if reprocess or not output_path.exists():
            to_process.append(pdf_path)

    if not to_process:
        console.print(f"[green]All {len(pdf_files)} PDFs already processed![/green]")
        return

    # Sort by file size descending -- largest PDFs first to minimize tail latency.
    to_process.sort(key=lambda p: p.stat().st_size, reverse=True)

    vm = psutil.virtual_memory()
    console.print(f"\n[bold green]Optimized Parallel OCR[/bold green]")
    console.print(f"[bold]Processing {len(to_process)} PDFs with {num_workers} workers[/bold]")
    console.print(f"[dim]Skipping {len(pdf_files) - len(to_process)} already processed[/dim]")
    console.print(
        f"[dim]System: {os.cpu_count()} cores, "
        f"{vm.total/(1024**3):.1f}GB RAM ({vm.percent:.0f}% used)[/dim]"
    )
    console.print(f"[dim]Optimizations: pool-init, ort-threads=1, torch-threads=1, size-desc sort[/dim]")
    console.print(f"[dim]Order: {to_process[0].name} ({to_process[0].stat().st_size//1024}KB) ... {to_process[-1].name} ({to_process[-1].stat().st_size//1024}KB)[/dim]\n")

    batch_start = time.time()
    start_timestamp = datetime.now().isoformat()

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_system, args=(10.0, stop_event), daemon=True
    )
    monitor_thread.start()

    pdf_paths_str = [str(p) for p in to_process]
    results = []

    init_start = time.time()
    with mp.Pool(processes=num_workers, initializer=_init_worker) as pool:
        init_time = time.time() - init_start
        console.print(f"[dim]Pool initialized in {init_time:.1f}s (converter loaded per worker)[/dim]\n")

        for i, result in enumerate(pool.imap_unordered(process_single_pdf, pdf_paths_str)):
            results.append(result)

            status = "[green]✓[/green]" if result["status"] == "success" else "[red]✗[/red]"
            vm = psutil.virtual_memory()

            if result["status"] == "success":
                console.print(
                    f"{status} [{i+1}/{len(to_process)}] {result['filename']}: "
                    f"{result['page_count']} pages, {result['block_count']} blocks, "
                    f"{result['table_count']} tables ({result['parse_time_seconds']:.1f}s) "
                    f"[dim]mem:{vm.percent:.0f}%[/dim]"
                )
            else:
                console.print(
                    f"{status} [{i+1}/{len(to_process)}] {result['filename']}: "
                    f"Error - {result['error'][:60]}... ({result['parse_time_seconds']:.1f}s)"
                )

            output_path = output_dir / f"{Path(result['filename']).stem}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    stop_event.set()
    batch_end = time.time()
    total_time = batch_end - batch_start

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    total_parse_time = sum(r["parse_time_seconds"] for r in results)

    avg_time = sum(r["parse_time_seconds"] for r in successful) / len(successful) if successful else 0
    total_pages = sum(r.get("page_count", 0) for r in successful)
    pdfs_per_hour = (3600 / avg_time) * num_workers * 0.90 if avg_time > 0 else 0

    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print("[bold green]Batch Complete (Optimized)[/bold green]")
    console.print(f"[bold]{'='*60}[/bold]")
    console.print(f"  Workers:        {num_workers}")
    console.print(f"  PDFs processed: {len(results)}")
    console.print(f"  Successful:     {len(successful)}")
    console.print(f"  Failed:         {len(failed)}")
    console.print(f"  Wall time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    console.print(f"  Total CPU time: {total_parse_time:.1f}s ({total_parse_time/60:.1f} min)")
    console.print(f"  Speedup:        {total_parse_time/total_time:.2f}x")
    console.print(f"  Avg per PDF:    {avg_time:.1f}s")
    console.print(f"  Total pages:    {total_pages}")
    console.print(f"  Est. PDFs/hr:   {pdfs_per_hour:.0f}")

    thrissur = 9294
    thrissur_hours = thrissur / pdfs_per_hour if pdfs_per_hour > 0 else float("inf")
    console.print(f"\n  Thrissur est:   {thrissur_hours:.0f}h ({thrissur_hours / 24:.1f} days)")

    summary = {
        "batch_start": start_timestamp,
        "batch_end": datetime.now().isoformat(),
        "script": "14_optimized_parallel_ocr.py",
        "optimizations": [
            "pool_initializer",
            "ort_threads_1",
            "torch_threads_1",
            "size_desc_sort",
            "pre_import_docling",
        ],
        "machine": {
            "cpu_count": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        },
        "num_workers": num_workers,
        "total_pdfs": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "wall_time_seconds": total_time,
        "total_cpu_time_seconds": total_parse_time,
        "speedup": total_parse_time / total_time if total_time > 0 else 0,
        "avg_time_per_pdf": avg_time,
        "pdfs_per_hour": pdfs_per_hour,
        "thrissur_est_hours": thrissur_hours,
        "results": [
            {
                "filename": r["filename"],
                "status": r["status"],
                "parse_time_seconds": r["parse_time_seconds"],
                "page_count": r.get("page_count"),
                "block_count": r.get("block_count"),
                "table_count": r.get("table_count"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }

    summary_path = output_dir / "_batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"\n[dim]Summary saved to {summary_path}[/dim]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Parallel Tesseract OCR with Docling")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--input-dir", type=Path, default=SAMPLE_PDFS_DIR)
    parser.add_argument("--output-dir", type=Path, default=SAMPLE_OUTPUT_DIR)
    parser.add_argument("--reprocess", action="store_true", help="Ignore already-processed files")
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.workers,
        reprocess=args.reprocess,
    )
