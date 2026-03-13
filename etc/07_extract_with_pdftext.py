#!/usr/bin/env python3
"""
Re-extract PDFs using pdftext (pypdfium2-based) for better Malayalam quality.

Saves output to sample_output2/ for comparison with Docling output in sample_output/
"""

import json
import time
from pathlib import Path

from rich.console import Console

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
SAMPLE_OUTPUT_DIR = Path(__file__).parent / "sample_output"
SAMPLE_OUTPUT2_DIR = Path(__file__).parent / "sample_output2"


def extract_with_pdftext(pdf_path: Path) -> dict:
    """Extract text using pdftext library."""
    from pdftext.extraction import plain_text_output, dictionary_output
    
    start = time.time()
    
    # Get plain text
    text = plain_text_output(str(pdf_path))
    
    # Get structured output (blocks)
    structured = dictionary_output(str(pdf_path))
    
    elapsed = time.time() - start
    
    # Count pages and blocks
    page_count = len(structured) if structured else 0
    block_count = sum(len(page.get("blocks", [])) for page in structured) if structured else 0
    
    return {
        "filename": pdf_path.name,
        "extractor": "pdftext",
        "parse_time_seconds": elapsed,
        "page_count": page_count,
        "block_count": block_count,
        "text": text,
        "structured": structured,
    }


def extract_with_pypdfium2(pdf_path: Path) -> dict:
    """Extract text using pypdfium2 directly."""
    import pypdfium2 as pdfium
    
    start = time.time()
    
    pdf = pdfium.PdfDocument(pdf_path)
    
    pages_text = []
    for page in pdf:
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        pages_text.append(text)
    
    pdf.close()
    
    elapsed = time.time() - start
    
    return {
        "filename": pdf_path.name,
        "extractor": "pypdfium2",
        "parse_time_seconds": elapsed,
        "page_count": len(pages_text),
        "text": "\n\n".join(pages_text),
        "pages": pages_text,
    }


def main():
    SAMPLE_OUTPUT2_DIR.mkdir(exist_ok=True)
    
    # Find PDFs that have corresponding JSON in sample_output
    existing_jsons = list(SAMPLE_OUTPUT_DIR.glob("*.json"))
    pdf_names = [j.stem + ".pdf" for j in existing_jsons]
    
    pdfs_to_process = [SAMPLE_PDFS_DIR / name for name in pdf_names if (SAMPLE_PDFS_DIR / name).exists()]
    
    console.print(f"[bold]Re-extracting {len(pdfs_to_process)} PDFs with pdftext/pypdfium2[/bold]\n")
    
    for pdf_path in pdfs_to_process:
        console.print(f"[dim]Processing {pdf_path.name}...[/dim]")
        
        # Extract with pypdfium2 (fastest, best fragmentation score)
        result = extract_with_pypdfium2(pdf_path)
        
        output_path = SAMPLE_OUTPUT2_DIR / f"{pdf_path.stem}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        console.print(f"  [green]Saved to {output_path.name}[/green] ({result['page_count']} pages, {result['parse_time_seconds']:.2f}s)")
    
    console.print(f"\n[bold green]Done! Output in {SAMPLE_OUTPUT2_DIR}[/bold green]")


if __name__ == "__main__":
    main()
