#!/usr/bin/env python3
"""
Chunking prototype for Docling-parsed PDFs.

Implements three chunk types per SPEC.md:
1. KV chunks - form fields (label: value pairs)
2. Narrative chunks - paragraphs/sections with sliding window
3. Table chunks - schema, rows, and optional summary

Usage:
    python etc/04_chunk_test.py [--max-files N] [--verbose]
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import tiktoken
from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel

console = Console()

SAMPLE_OUTPUT_DIR = Path(__file__).parent / "sample_output"
CHUNK_OUTPUT_DIR = Path(__file__).parent / "chunk_output"

KV_TARGET_TOKENS = 500
KV_MAX_FIELDS = 5
NARRATIVE_TARGET_TOKENS = 500
NARRATIVE_OVERLAP_TOKENS = 75
TABLE_ROWS_PER_CHUNK = 8  # Reduced to keep under 800 token limit
MAX_CHUNK_TOKENS = 800


@dataclass
class ChunkObject:
    """A single chunk ready for embedding."""
    chunk_type: str  # kv | narrative | table_schema | table_rows | table_summary
    section_path: list[str]
    display_text: str
    search_text: str
    page_start: int
    page_end: int
    metadata: dict = field(default_factory=dict)


class Tokenizer:
    """Token counter using tiktoken (cl100k_base for approximation)."""
    
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def count(self, text: str) -> int:
        return len(self.enc.encode(text))


def normalize_text(text: str) -> str:
    """Normalize text for search: NFC Unicode, strip extra whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_kv_pattern(text: str) -> bool:
    """
    Detect if text looks like a key-value field.
    Patterns:
    - "Label : Value" or "Label: Value"
    - "1. Field name" followed by value
    - Malayalam form fields often have numbered items
    """
    patterns = [
        r"^[\d]+[\.\)]\s*.+",  # Numbered items: "1. Field" or "1) Field"
        r"^.{2,50}\s*:\s*.+",  # Label: Value (label 2-50 chars)
        r"^[A-Z][\.\)]\s*.+",  # Lettered items: "A. Section"
    ]
    for pattern in patterns:
        if re.match(pattern, text.strip()):
            return True
    return False


def extract_kv_pair(text: str) -> tuple[str, str] | None:
    """Extract key-value pair from text if present."""
    match = re.match(r"^(.{2,80}?)\s*:\s*(.+)$", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    match = re.match(r"^([\d]+[\.\)])\s*(.+)$", text.strip())
    if match:
        return match.group(1), match.group(2).strip()
    
    return None


def classify_block(block: dict) -> str:
    """
    Classify a block as kv, narrative, or table.
    
    Block types from Docling:
    - TextItem: regular text
    - ListItem: list items (often KV in forms)
    - SectionHeaderItem: section headers
    - TableItem: tables (handled separately)
    - PictureItem: images (skip)
    """
    block_type = block.get("type", "")
    text = block.get("text", "")
    
    if block_type == "TableItem":
        return "table"
    
    if block_type == "PictureItem":
        return "skip"
    
    if block_type == "SectionHeaderItem":
        return "section_header"
    
    if block_type == "ListItem" or is_kv_pattern(text):
        return "kv"
    
    if len(text) > 100:
        return "narrative"
    
    if is_kv_pattern(text):
        return "kv"
    
    return "narrative"


class Chunker:
    """Main chunker that processes Docling output into chunks."""
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.current_section_path: list[str] = []
    
    def chunk_document(self, doc: dict) -> list[ChunkObject]:
        """Process a full document into chunks."""
        chunks = []
        blocks = doc.get("blocks", [])
        tables = doc.get("tables", [])
        
        kv_buffer = []
        narrative_buffer = []
        table_idx = 0
        
        for block in blocks:
            block_class = classify_block(block)
            
            if block_class == "section_header":
                if kv_buffer:
                    chunks.extend(self._flush_kv_buffer(kv_buffer))
                    kv_buffer = []
                if narrative_buffer:
                    chunks.extend(self._flush_narrative_buffer(narrative_buffer))
                    narrative_buffer = []
                
                header_text = block.get("text", "").strip()
                if header_text:
                    level = block.get("level", 1)
                    if level == 1:
                        self.current_section_path = [header_text]
                    elif level == 2 and self.current_section_path:
                        self.current_section_path = [self.current_section_path[0], header_text]
                    else:
                        self.current_section_path.append(header_text)
                continue
            
            if block_class == "table":
                if kv_buffer:
                    chunks.extend(self._flush_kv_buffer(kv_buffer))
                    kv_buffer = []
                if narrative_buffer:
                    chunks.extend(self._flush_narrative_buffer(narrative_buffer))
                    narrative_buffer = []
                
                if table_idx < len(tables):
                    table_data = tables[table_idx]
                    page = block.get("page", 1)
                    chunks.extend(self._chunk_table(table_data, page))
                    table_idx += 1
                continue
            
            if block_class == "skip":
                continue
            
            if block_class == "kv":
                if narrative_buffer:
                    chunks.extend(self._flush_narrative_buffer(narrative_buffer))
                    narrative_buffer = []
                kv_buffer.append(block)
            else:
                if kv_buffer:
                    chunks.extend(self._flush_kv_buffer(kv_buffer))
                    kv_buffer = []
                narrative_buffer.append(block)
        
        if kv_buffer:
            chunks.extend(self._flush_kv_buffer(kv_buffer))
        if narrative_buffer:
            chunks.extend(self._flush_narrative_buffer(narrative_buffer))
        
        return chunks
    
    def _flush_kv_buffer(self, blocks: list[dict]) -> list[ChunkObject]:
        """Convert accumulated KV blocks into chunks."""
        chunks = []
        current_fields = []
        current_tokens = 0
        page_start = blocks[0].get("page", 1) if blocks else 1
        page_end = page_start
        
        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue
            
            page = block.get("page", page_end)
            page_end = max(page_end, page)
            
            field_tokens = self.tokenizer.count(text)
            
            if (current_tokens + field_tokens > KV_TARGET_TOKENS or 
                len(current_fields) >= KV_MAX_FIELDS) and current_fields:
                chunk = self._create_kv_chunk(current_fields, page_start, page_end)
                chunks.append(chunk)
                current_fields = []
                current_tokens = 0
                page_start = page
            
            current_fields.append(text)
            current_tokens += field_tokens
        
        if current_fields:
            chunk = self._create_kv_chunk(current_fields, page_start, page_end)
            chunks.append(chunk)
        
        return chunks
    
    def _create_kv_chunk(self, fields: list[str], page_start: int, page_end: int) -> ChunkObject:
        """Create a KV chunk from a list of field texts."""
        section_header = " > ".join(self.current_section_path) if self.current_section_path else ""
        
        display_lines = []
        if section_header:
            display_lines.append(f"[{section_header}]")
        display_lines.extend(fields)
        display_text = "\n".join(display_lines)
        
        search_parts = []
        if section_header:
            search_parts.append(section_header + ".")
        for f in fields:
            kv = extract_kv_pair(f)
            if kv:
                search_parts.append(f"{kv[0]} {kv[1]}.")
            else:
                search_parts.append(f + ".")
        search_text = normalize_text(" ".join(search_parts))
        
        return ChunkObject(
            chunk_type="kv",
            section_path=list(self.current_section_path),
            display_text=display_text,
            search_text=search_text,
            page_start=page_start,
            page_end=page_end,
            metadata={"field_count": len(fields), "token_count": self.tokenizer.count(search_text)}
        )
    
    def _flush_narrative_buffer(self, blocks: list[dict]) -> list[ChunkObject]:
        """Convert accumulated narrative blocks into chunks with overlap."""
        chunks = []
        
        all_text = []
        pages = []
        for block in blocks:
            text = block.get("text", "").strip()
            if text:
                all_text.append(text)
                pages.append(block.get("page", 1))
        
        if not all_text:
            return chunks
        
        combined = " ".join(all_text)
        combined_tokens = self.tokenizer.count(combined)
        
        if combined_tokens <= NARRATIVE_TARGET_TOKENS:
            chunk = self._create_narrative_chunk(
                combined, 
                min(pages) if pages else 1, 
                max(pages) if pages else 1
            )
            chunks.append(chunk)
        else:
            sentences = re.split(r'(?<=[.!?।])\s+', combined)
            
            current_sentences = []
            current_tokens = 0
            chunk_start_page = min(pages) if pages else 1
            
            for sentence in sentences:
                sent_tokens = self.tokenizer.count(sentence)
                
                if current_tokens + sent_tokens > NARRATIVE_TARGET_TOKENS and current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunk = self._create_narrative_chunk(chunk_text, chunk_start_page, max(pages))
                    chunks.append(chunk)
                    
                    overlap_tokens = 0
                    overlap_sentences = []
                    for s in reversed(current_sentences):
                        s_tokens = self.tokenizer.count(s)
                        if overlap_tokens + s_tokens <= NARRATIVE_OVERLAP_TOKENS:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    
                    current_sentences = overlap_sentences
                    current_tokens = overlap_tokens
                
                current_sentences.append(sentence)
                current_tokens += sent_tokens
            
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunk = self._create_narrative_chunk(chunk_text, chunk_start_page, max(pages))
                chunks.append(chunk)
        
        return chunks
    
    def _create_narrative_chunk(self, text: str, page_start: int, page_end: int) -> ChunkObject:
        """Create a narrative chunk."""
        section_header = " > ".join(self.current_section_path) if self.current_section_path else ""
        
        display_lines = []
        if section_header:
            display_lines.append(f"[{section_header}]")
        display_lines.append(text)
        display_text = "\n".join(display_lines)
        
        search_text = normalize_text(f"{section_header}. {text}" if section_header else text)
        
        return ChunkObject(
            chunk_type="narrative",
            section_path=list(self.current_section_path),
            display_text=display_text,
            search_text=search_text,
            page_start=page_start,
            page_end=page_end,
            metadata={"token_count": self.tokenizer.count(search_text)}
        )
    
    def _chunk_table(self, table: dict, page: int) -> list[ChunkObject]:
        """Create chunks for a table: schema + rows (+ optional summary)."""
        chunks = []
        
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        total_rows = table.get("total_rows", len(rows))
        
        if isinstance(headers[0], int) if headers else False:
            headers = [f"Col{i+1}" for i in range(len(headers))]
        else:
            headers = [str(h) for h in headers]
        
        section_header = " > ".join(self.current_section_path) if self.current_section_path else ""
        
        schema_lines = []
        if section_header:
            schema_lines.append(f"Section: {section_header}")
        schema_lines.append(f"Columns: {' | '.join(headers)}")
        if rows:
            sample_row = rows[0]
            sample_pairs = [f"{headers[i]}={sample_row[i]}" for i in range(min(len(headers), len(sample_row)))]
            schema_lines.append(f"Sample: {', '.join(sample_pairs)}")
        
        schema_display = "\n".join(schema_lines)
        schema_search = normalize_text(" ".join(schema_lines))
        
        chunks.append(ChunkObject(
            chunk_type="table_schema",
            section_path=list(self.current_section_path),
            display_text=schema_display,
            search_text=schema_search,
            page_start=page,
            page_end=page,
            metadata={
                "column_count": len(headers),
                "total_rows": total_rows,
                "token_count": self.tokenizer.count(schema_search)
            }
        ))
        
        for chunk_start in range(0, len(rows), TABLE_ROWS_PER_CHUNK):
            chunk_end = min(chunk_start + TABLE_ROWS_PER_CHUNK, len(rows))
            chunk_rows = rows[chunk_start:chunk_end]
            
            row_lines = []
            row_lines.append(f"Table rows {chunk_start + 1}–{chunk_end}:")
            row_lines.append(f"Columns: {' | '.join(headers)}")
            
            for row in chunk_rows:
                pairs = [f"{headers[i]}={row[i]}" for i in range(min(len(headers), len(row)))]
                row_lines.append(", ".join(pairs))
            
            rows_display = "\n".join(row_lines)
            rows_search = normalize_text(" ".join(row_lines))
            
            chunks.append(ChunkObject(
                chunk_type="table_rows",
                section_path=list(self.current_section_path),
                display_text=rows_display,
                search_text=rows_search,
                page_start=page,
                page_end=page,
                metadata={
                    "row_start": chunk_start + 1,
                    "row_end": chunk_end,
                    "token_count": self.tokenizer.count(rows_search)
                }
            ))
        
        if total_rows > 50:
            summary = f"This table contains {total_rows} rows with columns: {', '.join(headers)}."
            if section_header:
                summary = f"{section_header}. {summary}"
            
            chunks.append(ChunkObject(
                chunk_type="table_summary",
                section_path=list(self.current_section_path),
                display_text=summary,
                search_text=normalize_text(summary),
                page_start=page,
                page_end=page,
                metadata={"total_rows": total_rows, "token_count": self.tokenizer.count(summary)}
            ))
        
        return chunks


def print_chunk_stats(chunks: list[ChunkObject], filename: str):
    """Print statistics about generated chunks."""
    console.print(f"\n[bold cyan]Chunks for {filename}[/bold cyan]")
    
    type_counts = {}
    type_tokens = {}
    for chunk in chunks:
        t = chunk.chunk_type
        type_counts[t] = type_counts.get(t, 0) + 1
        type_tokens[t] = type_tokens.get(t, 0) + chunk.metadata.get("token_count", 0)
    
    table = RichTable(title="Chunk Statistics")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Avg Tokens", justify="right")
    
    for t in sorted(type_counts.keys()):
        count = type_counts[t]
        tokens = type_tokens[t]
        avg = tokens // count if count > 0 else 0
        table.add_row(t, str(count), str(tokens), str(avg))
    
    total_chunks = len(chunks)
    total_tokens = sum(type_tokens.values())
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_chunks}[/bold]", 
                  f"[bold]{total_tokens}[/bold]", 
                  f"[bold]{total_tokens // total_chunks if total_chunks else 0}[/bold]")
    
    console.print(table)


def print_sample_chunks(chunks: list[ChunkObject], n: int = 3):
    """Print sample chunks of each type."""
    by_type = {}
    for chunk in chunks:
        if chunk.chunk_type not in by_type:
            by_type[chunk.chunk_type] = []
        by_type[chunk.chunk_type].append(chunk)
    
    for chunk_type, type_chunks in by_type.items():
        console.print(f"\n[bold yellow]Sample {chunk_type} chunks:[/bold yellow]")
        for i, chunk in enumerate(type_chunks[:n]):
            display_preview = chunk.display_text[:300] + "..." if len(chunk.display_text) > 300 else chunk.display_text
            console.print(Panel(
                display_preview,
                title=f"{chunk_type} #{i+1} (pages {chunk.page_start}-{chunk.page_end}, {chunk.metadata.get('token_count', '?')} tokens)",
                border_style="dim"
            ))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk Docling-parsed PDFs")
    parser.add_argument("--max-files", type=int, default=3, help="Max files to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show sample chunks")
    args = parser.parse_args()
    
    CHUNK_OUTPUT_DIR.mkdir(exist_ok=True)
    
    json_files = sorted(SAMPLE_OUTPUT_DIR.glob("*.json"))[:args.max_files]
    
    if not json_files:
        console.print("[red]No JSON files found in sample_output/[/red]")
        return
    
    console.print(f"[bold]Processing {len(json_files)} parsed documents...[/bold]\n")
    
    chunker = Chunker()
    all_stats = {
        "total_chunks": 0,
        "by_type": {},
        "files_processed": 0
    }
    
    for json_path in json_files:
        console.print(f"[dim]Processing {json_path.name}...[/dim]")
        
        with open(json_path) as f:
            doc = json.load(f)
        
        chunker.current_section_path = []
        chunks = chunker.chunk_document(doc)
        
        print_chunk_stats(chunks, json_path.name)
        
        if args.verbose:
            print_sample_chunks(chunks)
        
        output_path = CHUNK_OUTPUT_DIR / json_path.name.replace(".json", "_chunks.json")
        with open(output_path, "w") as f:
            json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
        console.print(f"[green]Saved to {output_path}[/green]")
        
        all_stats["total_chunks"] += len(chunks)
        all_stats["files_processed"] += 1
        for chunk in chunks:
            t = chunk.chunk_type
            all_stats["by_type"][t] = all_stats["by_type"].get(t, 0) + 1
    
    console.print("\n" + "=" * 60)
    console.print("[bold green]Summary across all files:[/bold green]")
    console.print(f"  Files processed: {all_stats['files_processed']}")
    console.print(f"  Total chunks: {all_stats['total_chunks']}")
    console.print(f"  By type: {all_stats['by_type']}")


if __name__ == "__main__":
    main()
