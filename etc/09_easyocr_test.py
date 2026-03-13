#!/usr/bin/env python3
"""
Test EasyOCR-based extraction for Malayalam PDFs using Docling.

EasyOCR can use GPU acceleration and may have different quality characteristics.
"""

import json
import time
from pathlib import Path

from rich.console import Console

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
SAMPLE_OUTPUT_EASYOCR_DIR = Path(__file__).parent / "sample_output_easyocr"

# Test with just one PDF first for quality comparison
TEST_PDF = "Block_Panc_Ernakula_83.pdf"


def get_converter():
    """Create and return the Docling converter with EasyOCR settings."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.datamodel.base_models import InputFormat
    
    # EasyOCR uses different language codes than Tesseract
    # Malayalam = 'ml' (not 'mal')
    ocr_options = EasyOcrOptions(
        force_full_page_ocr=True,
        lang=["ml", "en"],  # EasyOCR codes
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
    """Extract text using Docling with EasyOCR."""
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
        "ocr_engine": "easyocr",
        "ocr_languages": ["ml", "en"],
        "parse_time_seconds": parse_time,
        "page_count": doc.num_pages(),
        "block_count": len(blocks),
        "table_count": len(tables),
        "markdown": markdown,
        "blocks": blocks,
        "tables": tables,
    }


def main():
    import torch
    
    SAMPLE_OUTPUT_EASYOCR_DIR.mkdir(exist_ok=True)
    
    pdf_path = SAMPLE_PDFS_DIR / TEST_PDF
    
    if not pdf_path.exists():
        console.print(f"[red]PDF not found: {pdf_path}[/red]")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        console.print(f"[green]GPU available: {torch.cuda.get_device_name(0)}[/green]")
    else:
        console.print("[yellow]No GPU detected - running on CPU (will be slow)[/yellow]")
    
    console.print(f"\n[bold]Testing EasyOCR extraction on: {pdf_path.name}[/bold]\n")
    
    # Load converter
    console.print("[yellow]Loading Docling with EasyOCR...[/yellow]")
    load_start = time.time()
    converter = get_converter()
    console.print(f"[green]Loaded in {time.time() - load_start:.1f}s[/green]\n")
    
    try:
        console.print(f"[cyan]Processing {pdf_path.name}...[/cyan]")
        result = extract_with_ocr(converter, pdf_path)
        
        output_path = SAMPLE_OUTPUT_EASYOCR_DIR / f"{pdf_path.stem}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        console.print(f"\n[green]Saved to {output_path}[/green]")
        console.print(f"  Pages: {result['page_count']}")
        console.print(f"  Blocks: {result['block_count']}")
        console.print(f"  Tables: {result['table_count']}")
        console.print(f"  Parse time: {result['parse_time_seconds']:.1f}s")
        
        # Show sample text
        console.print("\n[bold yellow]Sample extracted text (first 1500 chars):[/bold yellow]")
        console.print(result["markdown"][:1500])
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
