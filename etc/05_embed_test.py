#!/usr/bin/env python3
"""
Embedding prototype using BGE-M3 on chunked documents.

Tests:
1. Model loading time
2. Single embedding generation
3. Batch embedding throughput
4. Vector dimensions and similarity search demo

Usage:
    python etc/05_embed_test.py [--max-chunks N] [--batch-size N]
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table as RichTable
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

CHUNK_OUTPUT_DIR = Path(__file__).parent / "chunk_output"
EMBED_OUTPUT_DIR = Path(__file__).parent / "embed_output"

BGE_M3_MODEL = "BAAI/bge-m3"
EXPECTED_DIM = 1024
DEFAULT_BATCH_SIZE = 16  # CPU-friendly batch size


@dataclass
class EmbedResult:
    """Result of embedding a chunk."""
    chunk_idx: int
    chunk_type: str
    text_preview: str
    vector: np.ndarray
    embed_time_ms: float


def load_bge_m3():
    """Load BGE-M3 model using FlagEmbedding."""
    console.print("[yellow]Loading BGE-M3 model...[/yellow]")
    console.print("[dim]This may take a minute on first run (downloading ~2GB model)[/dim]")
    
    start = time.time()
    
    from FlagEmbedding import BGEM3FlagModel
    
    model = BGEM3FlagModel(
        BGE_M3_MODEL,
        use_fp16=False,  # CPU mode
        device="cpu"
    )
    
    elapsed = time.time() - start
    console.print(f"[green]Model loaded in {elapsed:.1f}s[/green]\n")
    
    return model


def embed_single(model, text: str) -> tuple[np.ndarray, float]:
    """Embed a single text and return vector + time."""
    start = time.time()
    
    output = model.encode(
        [text],
        batch_size=1,
        max_length=512,  # BGE-M3 recommends ≤512 for retrieval
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    
    elapsed_ms = (time.time() - start) * 1000
    vector = output["dense_vecs"][0]
    
    return np.array(vector), elapsed_ms


def embed_batch(model, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[np.ndarray, float]:
    """Embed a batch of texts and return vectors + total time."""
    start = time.time()
    
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    
    elapsed_ms = (time.time() - start) * 1000
    vectors = np.array(output["dense_vecs"])
    
    return vectors, elapsed_ms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_chunks(max_chunks: int | None = None) -> list[dict]:
    """Load chunks from chunk_output directory."""
    chunks = []
    
    for json_path in sorted(CHUNK_OUTPUT_DIR.glob("*_chunks.json")):
        with open(json_path) as f:
            file_chunks = json.load(f)
        
        for chunk in file_chunks:
            chunk["source_file"] = json_path.stem
            chunks.append(chunk)
        
        if max_chunks and len(chunks) >= max_chunks:
            break
    
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    return chunks


def run_similarity_demo(model, chunks: list[dict], embeddings: np.ndarray):
    """Demo: find similar chunks to a query."""
    console.print("\n[bold cyan]Similarity Search Demo[/bold cyan]")
    
    test_queries = [
        "പപനജക ക ക budget allocation",  # Mixed Malayalam + English
        "stadium electrification",
        "ഗണപഭനകനകള beneficiaries",
        "construction work tender",
    ]
    
    for query in test_queries:
        console.print(f"\n[yellow]Query:[/yellow] {query}")
        
        query_vec, _ = embed_single(model, query)
        
        similarities = [cosine_similarity(query_vec, emb) for emb in embeddings]
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            chunk = chunks[idx]
            sim = similarities[idx]
            preview = chunk["search_text"][:100] + "..." if len(chunk["search_text"]) > 100 else chunk["search_text"]
            console.print(f"  [green]#{rank}[/green] (sim={sim:.3f}) [{chunk['chunk_type']}] {preview}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BGE-M3 embeddings on chunks")
    parser.add_argument("--max-chunks", type=int, default=50, help="Max chunks to embed")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for embedding")
    parser.add_argument("--skip-demo", action="store_true", help="Skip similarity demo")
    args = parser.parse_args()
    
    EMBED_OUTPUT_DIR.mkdir(exist_ok=True)
    
    chunks = load_chunks(args.max_chunks)
    if not chunks:
        console.print("[red]No chunks found. Run 04_chunk_test.py first.[/red]")
        return
    
    console.print(f"[bold]Loaded {len(chunks)} chunks for embedding test[/bold]\n")
    
    type_counts = {}
    for c in chunks:
        t = c["chunk_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    console.print(f"Chunk types: {type_counts}\n")
    
    model = load_bge_m3()
    
    console.print("[bold]Testing single embedding...[/bold]")
    test_text = chunks[0]["search_text"]
    vec, single_time = embed_single(model, test_text)
    
    console.print(f"  Vector shape: {vec.shape}")
    console.print(f"  Vector dtype: {vec.dtype}")
    console.print(f"  Time: {single_time:.1f}ms")
    
    if vec.shape[0] != EXPECTED_DIM:
        console.print(f"[red]WARNING: Expected {EXPECTED_DIM} dims, got {vec.shape[0]}[/red]")
    else:
        console.print(f"[green]  Dimension check passed ({EXPECTED_DIM})[/green]")
    
    console.print(f"\n[bold]Batch embedding {len(chunks)} chunks (batch_size={args.batch_size})...[/bold]")
    
    texts = [c["search_text"] for c in chunks]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Embedding...", total=1)
        
        embeddings, batch_time = embed_batch(model, texts, args.batch_size)
        
        progress.update(task, completed=1)
    
    throughput = len(chunks) / (batch_time / 1000)
    avg_time = batch_time / len(chunks)
    
    table = RichTable(title="Embedding Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Total chunks", str(len(chunks)))
    table.add_row("Total time", f"{batch_time:.1f}ms")
    table.add_row("Avg per chunk", f"{avg_time:.1f}ms")
    table.add_row("Throughput", f"{throughput:.1f} chunks/sec")
    table.add_row("Vector dimensions", str(embeddings.shape[1]))
    table.add_row("Memory (vectors)", f"{embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    console.print(table)
    
    console.print("\n[bold]Token count distribution:[/bold]")
    token_counts = [c["metadata"].get("token_count", 0) for c in chunks]
    console.print(f"  Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)//len(token_counts)}")
    
    if not args.skip_demo:
        run_similarity_demo(model, chunks, embeddings)
    
    output_data = {
        "model": BGE_M3_MODEL,
        "dimensions": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "embeddings": embeddings.tolist(),
        "chunks": [
            {
                "chunk_type": c["chunk_type"],
                "search_text": c["search_text"][:500],
                "source_file": c.get("source_file", "unknown")
            }
            for c in chunks
        ]
    }
    
    output_path = EMBED_OUTPUT_DIR / "embeddings_sample.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False)
    console.print(f"\n[green]Saved embeddings to {output_path}[/green]")
    
    console.print("\n[bold green]Embedding test complete![/bold green]")
    console.print(f"""
Summary:
- BGE-M3 produces {EXPECTED_DIM}-dimensional vectors
- CPU throughput: ~{throughput:.0f} chunks/sec (batch_size={args.batch_size})
- Estimated time for 80k PDFs (~100 chunks/PDF = 8M chunks):
  - CPU: ~{8_000_000 / throughput / 3600:.1f} hours
  - GPU (est. 10x faster): ~{8_000_000 / throughput / 3600 / 10:.1f} hours
""")


if __name__ == "__main__":
    main()
