#!/usr/bin/env python3
"""
Compare different PDF text extraction methods for Malayalam quality.

Tests:
1. pdftext (pypdfium2-based) - fast, modern
2. pdfplumber - layout-aware with tolerance tuning
3. PyMuPDF (fitz) - widely used
4. Direct pypdfium2 - low-level control

Usage:
    python etc/06_compare_extractors.py
"""

import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

TEST_PDF = Path(__file__).parent / "sample_pdfs" / "Block_Panc_Ernakula_83.pdf"


def extract_with_pdftext(pdf_path: Path) -> tuple[str, float]:
    """Extract using pdftext library."""
    start = time.time()
    try:
        from pdftext.extraction import plain_text_output
        
        text = plain_text_output(str(pdf_path))
        elapsed = time.time() - start
        return text, elapsed
    except Exception as e:
        return f"Error: {e}", time.time() - start


def extract_with_pdfplumber(pdf_path: Path, x_tolerance: int = 3) -> tuple[str, float]:
    """Extract using pdfplumber with layout mode."""
    start = time.time()
    try:
        import pdfplumber
        
        texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(
                    layout=True,
                    x_tolerance=x_tolerance,
                    y_tolerance=3
                )
                if text:
                    texts.append(text)
        
        elapsed = time.time() - start
        return "\n\n".join(texts), elapsed
    except Exception as e:
        return f"Error: {e}", time.time() - start


def extract_with_pymupdf(pdf_path: Path) -> tuple[str, float]:
    """Extract using PyMuPDF (fitz)."""
    start = time.time()
    try:
        import fitz
        
        texts = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text("text")
            texts.append(text)
        doc.close()
        
        elapsed = time.time() - start
        return "\n\n".join(texts), elapsed
    except Exception as e:
        return f"Error: {e}", time.time() - start


def extract_with_pypdfium2(pdf_path: Path) -> tuple[str, float]:
    """Extract using pypdfium2 directly."""
    start = time.time()
    try:
        import pypdfium2 as pdfium
        
        texts = []
        pdf = pdfium.PdfDocument(pdf_path)
        for page in pdf:
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            texts.append(text)
        pdf.close()
        
        elapsed = time.time() - start
        return "\n\n".join(texts), elapsed
    except Exception as e:
        return f"Error: {e}", time.time() - start


def analyze_malayalam_quality(text: str) -> dict:
    """Analyze text quality for Malayalam content."""
    import re
    
    malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]+')
    malayalam_words = malayalam_pattern.findall(text)
    
    single_char_words = sum(1 for w in malayalam_words if len(w) == 1)
    
    space_in_word = re.findall(r'[\u0D00-\u0D7F]\s[\u0D00-\u0D7F]', text)
    
    total_chars = len(text)
    total_spaces = text.count(' ')
    
    return {
        "total_chars": total_chars,
        "total_spaces": total_spaces,
        "space_ratio": total_spaces / total_chars if total_chars > 0 else 0,
        "malayalam_words": len(malayalam_words),
        "single_char_malayalam": single_char_words,
        "fragmentation_ratio": single_char_words / len(malayalam_words) if malayalam_words else 0,
        "mid_word_spaces": len(space_in_word),
    }


def show_sample(text: str, name: str, start_line: int = 5, num_lines: int = 10):
    """Show a sample of extracted text."""
    lines = text.split('\n')
    sample_lines = lines[start_line:start_line + num_lines]
    sample = '\n'.join(sample_lines)
    
    console.print(Panel(
        sample[:1000] if len(sample) > 1000 else sample,
        title=f"{name} - Lines {start_line+1}-{start_line+num_lines}",
        border_style="dim"
    ))


def main():
    if not TEST_PDF.exists():
        console.print(f"[red]Test PDF not found: {TEST_PDF}[/red]")
        return
    
    console.print(f"[bold]Comparing PDF extractors on: {TEST_PDF.name}[/bold]\n")
    
    extractors = [
        ("pdftext", lambda p: extract_with_pdftext(p)),
        ("pdfplumber (x_tol=3)", lambda p: extract_with_pdfplumber(p, x_tolerance=3)),
        ("pdfplumber (x_tol=1)", lambda p: extract_with_pdfplumber(p, x_tolerance=1)),
        ("PyMuPDF", lambda p: extract_with_pymupdf(p)),
        ("pypdfium2", lambda p: extract_with_pypdfium2(p)),
    ]
    
    results = []
    
    for name, extractor in extractors:
        console.print(f"[dim]Testing {name}...[/dim]")
        text, elapsed = extractor(TEST_PDF)
        
        if text.startswith("Error:"):
            console.print(f"[red]{name}: {text}[/red]")
            continue
        
        quality = analyze_malayalam_quality(text)
        quality["name"] = name
        quality["time"] = elapsed
        quality["text"] = text
        results.append(quality)
    
    table = Table(title="Extraction Quality Comparison")
    table.add_column("Extractor", style="cyan")
    table.add_column("Time (s)", justify="right")
    table.add_column("Total Chars", justify="right")
    table.add_column("Space Ratio", justify="right")
    table.add_column("Malayalam Words", justify="right")
    table.add_column("Fragmentation", justify="right")
    table.add_column("Mid-word Spaces", justify="right")
    
    for r in results:
        table.add_row(
            r["name"],
            f"{r['time']:.2f}",
            str(r["total_chars"]),
            f"{r['space_ratio']:.3f}",
            str(r["malayalam_words"]),
            f"{r['fragmentation_ratio']:.3f}",
            str(r["mid_word_spaces"]),
        )
    
    console.print(table)
    
    console.print("\n[bold]Quality Metrics Explained:[/bold]")
    console.print("  - Space Ratio: Lower is usually better (less fragmentation)")
    console.print("  - Fragmentation: Ratio of single-char Malayalam 'words' (lower = better)")
    console.print("  - Mid-word Spaces: Spaces between Malayalam chars (lower = better)")
    
    console.print("\n[bold yellow]Sample Output Comparison (Page 1, Lines 5-15):[/bold yellow]")
    for r in results:
        show_sample(r["text"], r["name"], start_line=0, num_lines=15)
    
    if results:
        best = min(results, key=lambda x: x["fragmentation_ratio"])
        console.print(f"\n[bold green]Best quality: {best['name']} (fragmentation: {best['fragmentation_ratio']:.3f})[/bold green]")


if __name__ == "__main__":
    main()
