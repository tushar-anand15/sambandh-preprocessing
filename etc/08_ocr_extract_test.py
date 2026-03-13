#!/usr/bin/env python3
"""
OCR-based extraction for Malayalam PDFs using Docling with force_full_page_ocr.

This bypasses the broken text layer and re-reads the visual text.
"""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
SAMPLE_OUTPUT_OCR_DIR = Path(__file__).parent / "sample_output_ocr"


def get_converter():
    """Create and return the Docling converter with OCR settings."""
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
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_with_ocr(converter, pdf_path: Path) -> dict:
    """Extract text using Docling with forced OCR via Tesseract CLI."""
    parse_start = time.time()
    
    result = converter.convert(str(pdf_path))
    doc = result.document
    
    parse_time = time.time() - parse_start
    
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
        tables.append({
            "headers": list(table_data.columns),
            "rows": table_data.values.tolist(),
            "total_rows": len(table_data),
        })
    
    return {
        "filename": pdf_path.name,
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


def main():
    SAMPLE_OUTPUT_OCR_DIR.mkdir(exist_ok=True)
    
    pdf_files = sorted(SAMPLE_PDFS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"[red]No PDFs found in {SAMPLE_PDFS_DIR}[/red]")
        return
    
    # Skip already processed files
    to_process = []
    for pdf_path in pdf_files:
        output_path = SAMPLE_OUTPUT_OCR_DIR / f"{pdf_path.stem}.json"
        if not output_path.exists():
            to_process.append(pdf_path)
    
    if not to_process:
        console.print(f"[green]All {len(pdf_files)} PDFs already processed![/green]")
        return
    
    console.print(f"[bold]Processing {len(to_process)} PDFs with OCR (skipping {len(pdf_files) - len(to_process)} already done)[/bold]\n")
    
    # Load converter once
    console.print("[yellow]Loading Docling with Tesseract OCR...[/yellow]")
    load_start = time.time()
    converter = get_converter()
    console.print(f"[green]Loaded in {time.time() - load_start:.1f}s[/green]\n")
    
    total_time = 0
    results_summary = []
    
    for i, pdf_path in enumerate(to_process, 1):
        console.print(f"[cyan][{i}/{len(to_process)}] Processing {pdf_path.name}...[/cyan]")
        
        try:
            result = extract_with_ocr(converter, pdf_path)
            
            output_path = SAMPLE_OUTPUT_OCR_DIR / f"{pdf_path.stem}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            total_time += result["parse_time_seconds"]
            results_summary.append({
                "file": pdf_path.name,
                "pages": result["page_count"],
                "blocks": result["block_count"],
                "tables": result["table_count"],
                "time": result["parse_time_seconds"],
            })
            
            console.print(f"  [green]Done: {result['page_count']} pages, {result['block_count']} blocks, {result['table_count']} tables in {result['parse_time_seconds']:.1f}s[/green]")
            
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    # Summary
    console.print(f"\n[bold green]Completed {len(results_summary)} PDFs in {total_time:.1f}s total[/bold green]")
    console.print(f"Average: {total_time/len(results_summary):.1f}s per PDF\n")
    
    console.print("[bold]Summary:[/bold]")
    for r in results_summary:
        console.print(f"  {r['file']}: {r['pages']} pages, {r['blocks']} blocks, {r['tables']} tables ({r['time']:.1f}s)")


if __name__ == "__main__":
    main()
