#!/usr/bin/env python3
"""
03_parse_test.py — Test PDF parsing with Docling on sample PDFs.

Runs Docling on each PDF in sample_pdfs/ and:
- Prints page count, block count, table count
- Shows first few blocks with type and text snippet
- Shows first table with headers and sample rows
- Saves full output to sample_output/<pdf_name>.json

Docling is faster than Marker on CPU and has excellent table extraction.
"""

import json
import os
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
SAMPLE_OUTPUT_DIR = Path(__file__).parent / "sample_output"

MAX_PDFS = int(os.getenv("MAX_PDFS", "5"))


def load_docling():
    """Load Docling converter."""
    console.print("[yellow]Loading Docling...[/yellow]")
    start = time.time()
    
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Disable OCR for born-digital PDFs (faster)
    pipeline_options.do_table_structure = True  # Enable table detection
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    elapsed = time.time() - start
    console.print(f"[green]Docling loaded in {elapsed:.1f}s[/green]\n")
    
    return converter


def parse_pdf(converter, pdf_path: Path) -> dict:
    """Parse a single PDF with Docling."""
    console.print(f"[dim]  Starting parse...[/dim]")
    start = time.time()
    
    try:
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        console.print(f"[red]  Converter error: {e}[/red]")
        raise
    
    elapsed = time.time() - start
    console.print(f"[dim]  Parse completed in {elapsed:.1f}s[/dim]")
    
    markdown = doc.export_to_markdown()
    
    blocks = []
    tables = []
    
    for item, level in doc.iterate_items():
        block_info = {
            "type": type(item).__name__,
            "level": level,
        }
        
        if hasattr(item, "text"):
            block_info["text"] = item.text[:500] if item.text else ""
        
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, "page_no"):
                block_info["page"] = prov.page_no
        
        blocks.append(block_info)
        
        if "Table" in type(item).__name__:
            table_data = extract_table_data(item)
            if table_data:
                tables.append(table_data)
    
    page_count = doc.num_pages() if hasattr(doc, "num_pages") else 0
    
    return {
        "filename": pdf_path.name,
        "parse_time_seconds": elapsed,
        "page_count": page_count,
        "markdown": markdown,
        "blocks": blocks,
        "tables": tables,
        "block_count": len(blocks),
        "table_count": len(tables),
    }


def extract_table_data(table_item) -> dict | None:
    """Extract structured data from a Docling table."""
    try:
        if hasattr(table_item, "export_to_dataframe"):
            df = table_item.export_to_dataframe()
            headers = list(df.columns)
            rows = df.values.tolist()[:10]
            return {
                "headers": headers,
                "rows": rows,
                "total_rows": len(df),
            }
        
        if hasattr(table_item, "data") and table_item.data:
            data = table_item.data
            if hasattr(data, "grid"):
                grid = data.grid
                headers = []
                rows = []
                for row_idx, row in enumerate(grid):
                    cells = [cell.text if hasattr(cell, "text") else str(cell) for cell in row]
                    if row_idx == 0:
                        headers = cells
                    else:
                        rows.append(cells)
                return {
                    "headers": headers,
                    "rows": rows[:10],
                    "total_rows": len(rows),
                }
        
        return None
    except Exception as e:
        console.print(f"[dim]  Table extraction note: {e}[/dim]")
        return None


def print_parse_result(result: dict):
    """Pretty print parse results."""
    console.rule(f"[bold blue]{result['filename']}")
    
    stats = Table(show_header=False, box=None)
    stats.add_column("Key", style="cyan")
    stats.add_column("Value", style="green")
    stats.add_row("Parse time", f"{result.get('parse_time_seconds', 0):.2f}s")
    stats.add_row("Pages", str(result.get("page_count", "?")))
    stats.add_row("Blocks", str(result.get("block_count", "?")))
    stats.add_row("Tables", str(result.get("table_count", "?")))
    console.print(stats)
    
    blocks = result.get("blocks", [])
    if blocks:
        console.print("\n[bold]First 5 Blocks:[/bold]")
        for i, block in enumerate(blocks[:5]):
            text_preview = block.get("text", "")[:100]
            if len(block.get("text", "")) > 100:
                text_preview += "..."
            page = block.get("page", "?")
            console.print(f"  [p{page}] [{block.get('type', 'Unknown')}] {text_preview}")
    
    tables = result.get("tables", [])
    if tables:
        console.print("\n[bold]First Table:[/bold]")
        t = tables[0]
        if t.get("headers"):
            console.print(f"  Headers: {t['headers'][:5]}")
        if t.get("rows") and len(t["rows"]) > 0:
            console.print(f"  Row 1: {t['rows'][0][:5] if t['rows'][0] else 'N/A'}")
            if len(t["rows"]) > 1:
                console.print(f"  Row 2: {t['rows'][1][:5] if t['rows'][1] else 'N/A'}")
        console.print(f"  Total rows: {t.get('total_rows', '?')}")
    
    md_preview = result.get("markdown", "")[:500]
    if md_preview:
        console.print("\n[bold]Markdown Preview:[/bold]")
        console.print(Panel(md_preview + ("..." if len(result.get("markdown", "")) > 500 else ""), 
                           title="First 500 chars", border_style="dim"))
    
    console.print()


def save_result(result: dict, output_dir: Path):
    """Save parse result to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = Path(result["filename"]).stem + ".json"
    output_path = output_dir / filename
    
    serializable = {
        "filename": result["filename"],
        "parse_time_seconds": result.get("parse_time_seconds"),
        "page_count": result.get("page_count"),
        "block_count": result.get("block_count"),
        "table_count": result.get("table_count"),
        "markdown": result.get("markdown", ""),
        "blocks": result.get("blocks", []),
        "tables": result.get("tables", []),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    return output_path


def get_output_path(pdf_path: Path) -> Path:
    """Get the output JSON path for a PDF."""
    return SAMPLE_OUTPUT_DIR / (pdf_path.stem + ".json")


def is_already_processed(pdf_path: Path) -> bool:
    """Check if PDF has already been processed (JSON exists and is valid)."""
    output_path = get_output_path(pdf_path)
    if not output_path.exists():
        return False
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("filename") == pdf_path.name and data.get("block_count", 0) > 0
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    console.print("[bold green]Docling PDF Parsing Test[/bold green]\n")
    console.print(f"[dim]MAX_PDFS={MAX_PDFS} (set MAX_PDFS env var to change)[/dim]\n")
    
    all_pdf_files = sorted(SAMPLE_PDFS_DIR.glob("*.pdf"))
    if not all_pdf_files:
        console.print(f"[red]No PDFs found in {SAMPLE_PDFS_DIR}[/red]")
        console.print("Run 02_download_samples.py first.")
        return
    
    already_done = [p for p in all_pdf_files if is_already_processed(p)]
    pending = [p for p in all_pdf_files if not is_already_processed(p)]
    
    console.print(f"Total PDFs: {len(all_pdf_files)}")
    console.print(f"[green]Already processed: {len(already_done)}[/green]")
    console.print(f"[yellow]Pending: {len(pending)}[/yellow]\n")
    
    if not pending:
        console.print("[green]All PDFs already processed![/green]")
        return
    
    pdf_files = pending[:MAX_PDFS]
    console.print(f"Processing {len(pdf_files)} PDFs this run\n")
    
    converter = load_docling()
    
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        console.print(f"[bold]Processing {i}/{len(pdf_files)}: {pdf_path.name}[/bold]")
        
        try:
            result = parse_pdf(converter, pdf_path)
            results.append(result)
            
            print_parse_result(result)
            
            output_path = save_result(result, SAMPLE_OUTPUT_DIR)
            console.print(f"[dim]Saved to {output_path}[/dim]\n")
            
        except Exception as e:
            console.print(f"[red]Error parsing {pdf_path.name}: {e}[/red]\n")
            import traceback
            traceback.print_exc()
            results.append({"filename": pdf_path.name, "error": str(e)})
    
    console.rule("[bold green]Summary")
    
    summary = Table(title="Parse Results Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    
    successful = [r for r in results if "error" not in r]
    summary.add_row("Processed this run", str(len(pdf_files)))
    summary.add_row("Successful", str(len(successful)))
    summary.add_row("Failed", str(len(results) - len(successful)))
    summary.add_row("Previously done", str(len(already_done)))
    summary.add_row("Total complete", str(len(already_done) + len(successful)))
    summary.add_row("Remaining", str(len(pending) - len(pdf_files)))
    
    if successful:
        avg_time = sum(r.get("parse_time_seconds", 0) for r in successful) / len(successful)
        total_blocks = sum(r.get("block_count", 0) for r in successful)
        total_tables = sum(r.get("table_count", 0) for r in successful)
        total_pages = sum(r.get("page_count", 0) for r in successful)
        
        summary.add_row("Avg parse time", f"{avg_time:.2f}s")
        summary.add_row("Total pages", str(total_pages))
        summary.add_row("Total blocks", str(total_blocks))
        summary.add_row("Total tables", str(total_tables))
        
        if total_pages > 0:
            pages_per_sec = total_pages / sum(r.get("parse_time_seconds", 1) for r in successful)
            summary.add_row("Pages/second", f"{pages_per_sec:.2f}")
    
    console.print(summary)
    console.print(f"\n[bold green]Output saved to {SAMPLE_OUTPUT_DIR}[/bold green]")


if __name__ == "__main__":
    main()
