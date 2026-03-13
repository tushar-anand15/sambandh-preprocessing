#!/usr/bin/env python3
"""
Parallel Tesseract OCR extraction for Malayalam PDFs using Docling.

Uses multiprocessing to process multiple PDFs in parallel.
Limits Tesseract to single-threaded mode to avoid CPU contention.
"""

import os

os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import sys
import time
import threading
import multiprocessing as mp
import psutil
from pathlib import Path
from datetime import datetime

from rich.console import Console

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
SAMPLE_OUTPUT_DIR = Path(__file__).parent / "sample_output_parallel"
NUM_WORKERS = 8


def get_converter():
    """Create and return the Docling converter with Tesseract OCR settings."""
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
    """Process a single PDF with Tesseract OCR. Runs in worker process."""
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


def worker_process(pdf_path_str: str) -> dict:
    return process_single_pdf(Path(pdf_path_str))


def monitor_system(interval: float = 10.0, stop_event: threading.Event = None):
    """Background thread: log load, CPU, memory every interval seconds."""
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
    num_workers: int = NUM_WORKERS,
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

    vm = psutil.virtual_memory()
    console.print(f"[bold]Processing {len(to_process)} PDFs with {num_workers} parallel workers[/bold]")
    console.print(f"[dim]Skipping {len(pdf_files) - len(to_process)} already processed[/dim]")
    console.print(
        f"[dim]System: {os.cpu_count()} cores, "
        f"{vm.total/(1024**3):.1f}GB RAM ({vm.percent:.0f}% used)[/dim]\n"
    )

    batch_start = time.time()
    start_timestamp = datetime.now().isoformat()

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_system, args=(10.0, stop_event), daemon=True
    )
    monitor_thread.start()

    pdf_paths_str = [str(p) for p in to_process]
    results = []

    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_process, pdf_paths_str)):
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
    console.print("[bold green]Batch Complete[/bold green]")
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
    console.print(f"\n  Thrissur est:   {thrissur / pdfs_per_hour:.0f}h ({thrissur / pdfs_per_hour / 24:.1f} days)")

    summary = {
        "batch_start": start_timestamp,
        "batch_end": datetime.now().isoformat(),
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

    parser = argparse.ArgumentParser(description="Parallel Tesseract OCR with Docling")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
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
