"""
Chunker module for processing Docling-parsed PDFs into chunks.

Implements three chunk types per SPEC.md:
1. KV chunks - form fields (label: value pairs)
2. Narrative chunks - paragraphs/sections with sliding window
3. Table chunks - schema, rows, and optional summary
"""

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import NamedTuple

import tiktoken

KV_TARGET_TOKENS = 500
KV_MAX_FIELDS = 5
NARRATIVE_TARGET_TOKENS = 500
NARRATIVE_OVERLAP_TOKENS = 75
TABLE_ROWS_PER_CHUNK = 8
MAX_TABLE_CHUNK_TOKENS = 3000


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

    def to_dict(self) -> dict:
        return asdict(self)


class ChunkStats(NamedTuple):
    """Statistics about chunking results."""

    chunk_count: int
    kv_chunks: int
    narrative_chunks: int
    table_schema_chunks: int
    table_rows_chunks: int
    table_summary_chunks: int
    total_tokens: int


class ChunkResult(NamedTuple):
    """Result of chunking a document."""

    chunks: list[ChunkObject]
    stats: ChunkStats


class Tokenizer:
    """Token counter using tiktoken (cl100k_base for approximation)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.enc = tiktoken.get_encoding("cl100k_base")
        return cls._instance

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

    def __init__(
        self,
        kv_target_tokens: int = KV_TARGET_TOKENS,
        kv_max_fields: int = KV_MAX_FIELDS,
        narrative_target_tokens: int = NARRATIVE_TARGET_TOKENS,
        narrative_overlap_tokens: int = NARRATIVE_OVERLAP_TOKENS,
        table_rows_per_chunk: int = TABLE_ROWS_PER_CHUNK,
        max_table_chunk_tokens: int = MAX_TABLE_CHUNK_TOKENS,
    ):
        self.tokenizer = Tokenizer()
        self.current_section_path: list[str] = []
        self.kv_target_tokens = kv_target_tokens
        self.kv_max_fields = kv_max_fields
        self.narrative_target_tokens = narrative_target_tokens
        self.narrative_overlap_tokens = narrative_overlap_tokens
        self.table_rows_per_chunk = table_rows_per_chunk
        self.max_table_chunk_tokens = max_table_chunk_tokens

    def chunk_document(self, doc: dict) -> ChunkResult:
        """Process a full document into chunks with statistics."""
        self.current_section_path = []
        chunks = self._process_document(doc)
        stats = self._compute_stats(chunks)
        return ChunkResult(chunks=chunks, stats=stats)

    def _compute_stats(self, chunks: list[ChunkObject]) -> ChunkStats:
        """Compute statistics from chunks."""
        type_counts = {}
        total_tokens = 0

        for chunk in chunks:
            t = chunk.chunk_type
            type_counts[t] = type_counts.get(t, 0) + 1
            total_tokens += chunk.metadata.get("token_count", 0)

        return ChunkStats(
            chunk_count=len(chunks),
            kv_chunks=type_counts.get("kv", 0),
            narrative_chunks=type_counts.get("narrative", 0),
            table_schema_chunks=type_counts.get("table_schema", 0),
            table_rows_chunks=type_counts.get("table_rows", 0),
            table_summary_chunks=type_counts.get("table_summary", 0),
            total_tokens=total_tokens,
        )

    def _process_document(self, doc: dict) -> list[ChunkObject]:
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

            if (
                current_tokens + field_tokens > self.kv_target_tokens
                or len(current_fields) >= self.kv_max_fields
            ) and current_fields:
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
            metadata={"field_count": len(fields), "token_count": self.tokenizer.count(search_text)},
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

        if combined_tokens <= self.narrative_target_tokens:
            chunk = self._create_narrative_chunk(
                combined, min(pages) if pages else 1, max(pages) if pages else 1
            )
            chunks.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?।])\s+", combined)

            current_sentences = []
            current_tokens = 0
            chunk_start_page = min(pages) if pages else 1

            for sentence in sentences:
                sent_tokens = self.tokenizer.count(sentence)

                if (
                    current_tokens + sent_tokens > self.narrative_target_tokens
                    and current_sentences
                ):
                    chunk_text = " ".join(current_sentences)
                    chunk = self._create_narrative_chunk(chunk_text, chunk_start_page, max(pages))
                    chunks.append(chunk)

                    overlap_tokens = 0
                    overlap_sentences = []
                    for s in reversed(current_sentences):
                        s_tokens = self.tokenizer.count(s)
                        if overlap_tokens + s_tokens <= self.narrative_overlap_tokens:
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
            metadata={"token_count": self.tokenizer.count(search_text)},
        )

    def _chunk_table(self, table: dict, page: int) -> list[ChunkObject]:
        """Create chunks for a table: schema + rows (+ optional summary).

        Row chunks are split by self.table_rows_per_chunk initially, then
        further split if the resulting chunk exceeds self.max_table_chunk_tokens.
        Each chunk repeats the column headers so it is self-contained.
        """
        chunks = []

        headers = table.get("headers", [])
        rows = table.get("rows", [])
        total_rows = table.get("total_rows", len(rows))

        if headers and isinstance(headers[0], int):
            headers = [f"Col{i+1}" for i in range(len(headers))]
        else:
            headers = [str(h) for h in headers]

        if not headers:
            return chunks

        section_header = " > ".join(self.current_section_path) if self.current_section_path else ""
        header_line = f"Columns: {' | '.join(headers)}"

        # --- Schema chunk (with sample row truncated to fit token limit) ---
        schema_lines = []
        if section_header:
            schema_lines.append(f"Section: {section_header}")
        schema_lines.append(header_line)
        if rows:
            sample_row = rows[0]
            sample_pairs = [
                f"{headers[i]}={sample_row[i]}" for i in range(min(len(headers), len(sample_row)))
            ]
            # Truncate sample pairs until schema fits within token limit
            while sample_pairs:
                candidate = normalize_text(
                    " ".join(schema_lines + [f"Sample: {', '.join(sample_pairs)}"])
                )
                if self.tokenizer.count(candidate) <= self.max_table_chunk_tokens:
                    schema_lines.append(f"Sample: {', '.join(sample_pairs)}")
                    break
                sample_pairs = sample_pairs[:-1]

        schema_display = "\n".join(schema_lines)
        schema_search = normalize_text(" ".join(schema_lines))

        chunks.append(
            ChunkObject(
                chunk_type="table_schema",
                section_path=list(self.current_section_path),
                display_text=schema_display,
                search_text=schema_search,
                page_start=page,
                page_end=page,
                metadata={
                    "column_count": len(headers),
                    "total_rows": total_rows,
                    "token_count": self.tokenizer.count(schema_search),
                },
            )
        )

        # --- Row chunks: token-aware accumulator ---
        # header_line is repeated in every row chunk so each chunk is self-contained
        header_tokens = self.tokenizer.count(header_line)
        current_rows: list[list] = []
        current_tokens = header_tokens
        chunk_first_row = 1  # 1-based index of first row in current buffer

        def _flush_rows(buffered_rows: list[list], first_row: int) -> ChunkObject:
            last_row = first_row + len(buffered_rows) - 1
            row_lines = [
                f"Table rows {first_row}–{last_row}:",
                header_line,
            ]
            for row in buffered_rows:
                pairs = [f"{headers[i]}={row[i]}" for i in range(min(len(headers), len(row)))]
                row_lines.append(", ".join(pairs))
            display = "\n".join(row_lines)
            search = normalize_text(" ".join(row_lines))
            return ChunkObject(
                chunk_type="table_rows",
                section_path=list(self.current_section_path),
                display_text=display,
                search_text=search,
                page_start=page,
                page_end=page,
                metadata={
                    "row_start": first_row,
                    "row_end": last_row,
                    "column_count": len(headers),
                    "total_rows": total_rows,
                    "token_count": self.tokenizer.count(search),
                },
            )

        for abs_idx, row in enumerate(rows):
            pairs = [f"{headers[i]}={row[i]}" for i in range(min(len(headers), len(row)))]
            row_text = ", ".join(pairs)
            row_tokens = self.tokenizer.count(row_text)

            should_flush = (
                current_rows
                and (
                    current_tokens + row_tokens > self.max_table_chunk_tokens
                    or len(current_rows) >= self.table_rows_per_chunk
                )
            )

            if should_flush:
                chunks.append(_flush_rows(current_rows, chunk_first_row))
                current_rows = []
                current_tokens = header_tokens
                chunk_first_row = abs_idx + 1  # next row is 1-based

            current_rows.append(row)
            current_tokens += row_tokens

        if current_rows:
            chunks.append(_flush_rows(current_rows, chunk_first_row))

        # --- Summary chunk for large tables ---
        if total_rows > 50:
            summary = f"This table contains {total_rows} rows with columns: {', '.join(headers)}."
            if section_header:
                summary = f"{section_header}. {summary}"

            chunks.append(
                ChunkObject(
                    chunk_type="table_summary",
                    section_path=list(self.current_section_path),
                    display_text=summary,
                    search_text=normalize_text(summary),
                    page_start=page,
                    page_end=page,
                    metadata={
                        "total_rows": total_rows,
                        "token_count": self.tokenizer.count(summary),
                    },
                )
            )

        return chunks
