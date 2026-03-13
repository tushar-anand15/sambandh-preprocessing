#!/usr/bin/env python3
"""
Test Surya OCR for Malayalam PDFs.

Surya is a modern ML-based OCR that supports 90+ languages including Malayalam,
with GPU acceleration.
"""

import json
import time
from pathlib import Path

from rich.console import Console

console = Console()

SAMPLE_PDFS_DIR = Path(__file__).parent / "sample_pdfs"
SAMPLE_OUTPUT_SURYA_DIR = Path(__file__).parent / "sample_output_surya"

# Test with just one PDF first
TEST_PDF = "Block_Panc_Ernakula_83.pdf"


def extract_with_surya(pdf_path: Path) -> dict:
    """Extract text from PDF using Surya OCR (new API)."""
    import torch
    from pdf2image import convert_from_path
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor
    from surya.foundation import FoundationPredictor
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Using device: {device}[/cyan]")
    
    # Load models
    console.print("[yellow]Loading Surya models...[/yellow]")
    model_start = time.time()
    
    det_predictor = DetectionPredictor(device=device)
    foundation_predictor = FoundationPredictor(device=device)
    rec_predictor = RecognitionPredictor(foundation_predictor)
    
    console.print(f"[green]Models loaded in {time.time() - model_start:.1f}s[/green]")
    
    # Convert PDF to images
    console.print("[yellow]Converting PDF to images...[/yellow]")
    img_start = time.time()
    pil_images = convert_from_path(str(pdf_path), dpi=200)
    console.print(f"[green]Converted {len(pil_images)} pages in {time.time() - img_start:.1f}s[/green]")
    
    # Run OCR (detection + recognition in one call)
    console.print("[yellow]Running OCR (detection + recognition)...[/yellow]")
    ocr_start = time.time()
    
    # New API: pass det_predictor, it handles everything
    rec_results = rec_predictor(
        pil_images,
        det_predictor=det_predictor,
    )
    
    ocr_time = time.time() - ocr_start
    total_time = time.time() - model_start
    console.print(f"[green]OCR completed in {ocr_time:.1f}s[/green]")
    
    # Extract text from results
    all_text = []
    blocks = []
    
    for page_idx, page_result in enumerate(rec_results):
        page_text_lines = []
        for line in page_result.text_lines:
            page_text_lines.append(line.text)
            blocks.append({
                "type": "TextLine",
                "text": line.text,
                "page": page_idx + 1,
                "confidence": getattr(line, 'confidence', None),
                "bbox": line.bbox if hasattr(line, 'bbox') else None,
            })
        all_text.append("\n".join(page_text_lines))
    
    full_text = "\n\n---\n\n".join(all_text)
    
    return {
        "filename": pdf_path.name,
        "extractor": "surya",
        "ocr_engine": "surya",
        "ocr_languages": ["ml", "en"],
        "device": device,
        "parse_time_seconds": total_time,
        "page_count": len(pil_images),
        "block_count": len(blocks),
        "table_count": 0,  # Surya doesn't do table detection
        "markdown": full_text,
        "blocks": blocks,
        "tables": [],
    }


def main():
    import torch
    
    SAMPLE_OUTPUT_SURYA_DIR.mkdir(exist_ok=True)
    
    pdf_path = SAMPLE_PDFS_DIR / TEST_PDF
    
    if not pdf_path.exists():
        console.print(f"[red]PDF not found: {pdf_path}[/red]")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        console.print(f"[green]GPU available: {torch.cuda.get_device_name(0)}[/green]")
    else:
        console.print("[yellow]No GPU detected - running on CPU (will be slower)[/yellow]")
    
    console.print(f"\n[bold]Testing Surya OCR on: {pdf_path.name}[/bold]\n")
    
    try:
        result = extract_with_surya(pdf_path)
        
        output_path = SAMPLE_OUTPUT_SURYA_DIR / f"{pdf_path.stem}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        console.print(f"\n[green]Saved to {output_path}[/green]")
        console.print(f"  Device: {result['device']}")
        console.print(f"  Pages: {result['page_count']}")
        console.print(f"  Text blocks: {result['block_count']}")
        console.print(f"  Total time: {result['parse_time_seconds']:.1f}s")
        
        # Show sample text
        console.print("\n[bold yellow]Sample extracted text (first 1500 chars):[/bold yellow]")
        console.print(result["markdown"][:1500])
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
