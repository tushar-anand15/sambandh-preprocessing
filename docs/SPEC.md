# Sambandh — RAG Retrieval Pipeline
## Product Requirements + Technical Specification (OSS-only)

> **Repo:** `sulekha-preprocessing`  
> **Status:** Draft v0.1  
> **Last updated:** 2026-03-12

---

## Table of Contents

1. [Objective and Scope](#1-objective-and-scope)
2. [System Context](#2-system-context)
3. [Architecture Overview](#3-architecture-overview)
4. [Ingestion Pipeline — Detailed Flow](#4-ingestion-pipeline--detailed-flow)
5. [Query Pipeline — Detailed Flow](#5-query-pipeline--detailed-flow)
6. [Data Model (RAG Postgres)](#6-data-model-rag-postgres)
7. [Parsing and Document Intelligence](#7-parsing-and-document-intelligence)
8. [Chunking Specification](#8-chunking-specification)
9. [Embeddings and Reranking](#9-embeddings-and-reranking)
10. [Hybrid Retrieval and Fusion](#10-hybrid-retrieval-and-fusion)
11. [Service API Contracts](#11-service-api-contracts)
12. [Directory Structure](#12-directory-structure)
13. [Infrastructure and Deployment](#13-infrastructure-and-deployment)
14. [Idempotency and Versioning](#14-idempotency-and-versioning)
15. [Observability](#15-observability)
16. [Evaluation Plan](#16-evaluation-plan)
17. [Milestones](#17-milestones)
18. [Open Questions](#18-open-questions)

---

## 1. Objective and Scope

Build a production-grade retrieval pipeline for **Sulekha PDFs** (Kerala municipal project documents — Malayalam + English), including:

- PDF ingestion from GCS (scraped by the existing `sulekha-scraper` service)
- Layout-preserving parsing with table and form extraction
- Structured chunking suitable for multilingual retrieval
- Hybrid indexing: `pgvector` (dense ANN) + Postgres FTS (lexical)
- Hybrid recall + RRF fusion + reranking
- A query API returning ranked chunks with page-level citations and full project metadata

### In Scope

| Capability | Notes |
|---|---|
| PDF ingestion from GCS | Read-only access to scraper bucket |
| Metadata enrichment | Read-only join to scraper Postgres |
| Parsing (layout + tables + OCR fallback) | Marker + Surya |
| Chunking | KV, narrative, table schema/rows/summary |
| Embedding | OSS multilingual (BGE-M3 default) |
| Indexing | pgvector HNSW + Postgres FTS (tsvector) |
| Hybrid retrieval | Dense + FTS → RRF → rerank |
| Citations | Page-level provenance per chunk |
| Ingestion status API | Per-PDF status, stage, versions |
| Evaluation harness | Malayalam + English + table queries |
| Observability | Structured logs, stage metrics |
| Versioned reprocessing | On model/parser/chunker upgrade |

### Out of Scope (v1)

- LLM generation / question-answering layer
- Human annotation UI
- Normalized domain-specific table extraction (e.g. structured project budget DB)
- Multi-tenant auth
- Sakarma meeting PDFs *(deferred to v1.5)*

---

## 2. System Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       External Systems (read-only)                      │
│                                                                         │
│   sulekha-scraper                                                       │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Scraper Postgres                GCS Bucket: sulekha-pdfs        │  │
│   │  ┌──────────────┐               ┌─────────────────────────────┐  │  │
│   │  │ pdfs         │               │ pdfs/{year}/{lb_type}/      │  │  │
│   │  │ projects     │               │   {district}/{lb}/{no}.pdf  │  │  │
│   │  │ local_bodies │               └─────────────────────────────┘  │  │
│   │  │ districts    │                                                 │  │
│   │  └──────────────┘                                                 │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                  │ read-only poll               │ download bytes
                  ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   sulekha-preprocessing (this repo)                     │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Ingestion Service                                                 │ │
│  │  • Polls scraper DB for pdfs.status = UPLOADED                     │ │
│  │  • Downloads PDF bytes from GCS                                    │ │
│  │  • Parses → chunks → embeds → indexes                              │ │
│  └───────────────────────────┬────────────────────────────────────────┘ │
│                              │ writes                                   │
│  ┌───────────────────────────▼────────────────────────────────────────┐ │
│  │  RAG Postgres (pgvector)                                           │ │
│  │  rag_pdfs | rag_chunks | rag_blocks | rag_tables | rag_runs        │ │
│  └───────────────────────────▲────────────────────────────────────────┘ │
│                              │ reads                                    │
│  ┌───────────────────────────┴────────────────────────────────────────┐ │
│  │  Query Service                                                     │ │
│  │  • Dense + FTS recall → RRF → rerank                               │ │
│  │  • Returns ranked chunks + citations + rag_metadata                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Network assumption

Both the scraper stack and this service are deployed on the **same Docker network** (`sulekha-network`). The scraper Postgres is reachable via hostname `scraper-postgres` (or `sulekha-postgres` depending on container name). No shared compose files are needed; this service simply consumes the scraper's network at runtime.

---

## 3. Architecture Overview

### Component diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  sulekha-preprocessing                                                      │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐  │
│  │   ingestion-worker   │    │   query-api  (FastAPI)                   │  │
│  │                      │    │                                          │  │
│  │  Poller              │    │  POST /search                            │  │
│  │    │                 │    │    │                                     │  │
│  │    ▼                 │    │    ├─► dense_recall (pgvector ANN)       │  │
│  │  fetch_new_pdfs()    │    │    ├─► fts_recall (tsvector GIN)         │  │
│  │    │                 │    │    ├─► rrf_fuse()                        │  │
│  │    ▼                 │    │    └─► rerank (cross-encoder)            │  │
│  │  [job queue Redis]   │    │                                          │  │
│  │    │                 │    │  GET /ingest/status?pdf_id=...           │  │
│  │    ▼                 │    └──────────────────────────────────────────┘  │
│  │  Stage 1: download   │                                                  │
│  │  Stage 2: parse      │    ┌──────────────────────────────────────────┐  │
│  │  Stage 3: chunk      │    │   RAG Postgres  (postgres:16 + pgvector) │  │
│  │  Stage 4: embed      │    │                                          │  │
│  │  Stage 5: index      │◄──►│   rag_pdfs        rag_chunks             │  │
│  │                      │    │   rag_blocks       rag_tables            │  │
│  └──────────────────────┘    │   rag_ingestion_runs                     │  │
│                              └──────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐  │
│  │   Redis              │    │   Model Cache (local FS / volume)        │  │
│  │   (job queue)        │    │   BGE-M3, jina-reranker, Surya, Marker   │  │
│  └──────────────────────┘    └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

External (read-only access):
  scraper-postgres:5432/sulekha  ──►  pdfs, projects, local_bodies, districts
  GCS gs://sulekha-pdfs/         ──►  pdfs/{year}/{lb_type}/{district}/{lb}/{no}.pdf
```

---

## 4. Ingestion Pipeline — Detailed Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                                   │
│                                                                               │
│  ┌────────────┐                                                               │
│  │   Poller   │  Every N minutes (configurable; default 5m)                  │
│  │            │  SELECT p.id, p.gcs_path, p.gcs_bucket,                      │
│  │            │         p.content_hash, pr.project_no,                       │
│  │            │         pr.project_name, pr.formulation, pr.expense,         │
│  │            │         lb.lb_name, d.district_name,                         │
│  │            │         d.year_label, d.lb_type_label                        │
│  │            │  FROM pdfs p                                                  │
│  │            │  JOIN projects pr ON pr.id = p.project_id                    │
│  │            │  JOIN local_bodies lb ON lb.id = pr.local_body_id            │
│  │            │  JOIN districts d ON d.id = lb.district_id                   │
│  │            │  WHERE p.status = 'UPLOADED'                                 │
│  │            │    AND p.uploaded_at > :checkpoint  -- or NOT IN rag_pdfs    │
│  └─────┬──────┘                                                               │
│        │ enqueue                                                              │
│        ▼                                                                      │
│  ┌─────────────┐   idempotency check:                                         │
│  │  Job Queue  │   if (pdf_id, content_hash, parse_v, chunk_v, embed_v)      │
│  │  (Redis)    │   already in rag_pdfs AND status=INDEXED → skip             │
│  └─────┬───────┘                                                               │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  Stage 1: DOWNLOAD                                                      │  │
│  │  • GCS download(gcs_bucket, gcs_path) → bytes                          │  │
│  │  • verify SHA256 == content_hash                                        │  │
│  │  • write rag_pdfs row (status=DOWNLOADING, versions, rag_metadata)     │  │
│  └──────────────────────────────┬──────────────────────────────────────────┘  │
│                                 │                                              │
│  ┌──────────────────────────────▼──────────────────────────────────────────┐  │
│  │  Stage 2: PARSE  (parse_version = "marker-v1.0")                        │  │
│  │                                                                          │  │
│  │  path: born-digital                                                      │  │
│  │    Marker.convert(pdf_bytes)                                             │  │
│  │      → {"blocks": [...], "tables": [...], "pages": [...]}                │  │
│  │                                                                          │  │
│  │  fallback: OCR (if quality_score < threshold OR is_scanned)              │  │
│  │    Surya.ocr(pdf_bytes) → layout blocks                                  │  │
│  │    Surya.detect_tables() → table JSON                                    │  │
│  │                                                                          │  │
│  │  output: ParseResult{blocks, tables, page_count, is_ocr, quality_score} │  │
│  │  writes: rag_blocks[], rag_tables[]  (status → PARSED)                   │  │
│  └──────────────────────────────┬──────────────────────────────────────────┘  │
│                                 │                                              │
│  ┌──────────────────────────────▼──────────────────────────────────────────┐  │
│  │  Stage 3: CHUNK  (chunk_version = "v1.0")                               │  │
│  │                                                                          │  │
│  │  chunker.run(parse_result, rag_metadata)                                 │  │
│  │    → for each block in reading order:                                    │  │
│  │        classify_block() → kv | narrative | table                         │  │
│  │        if kv:        KVChunker     → 1-5 fields/chunk, ~500 tokens       │  │
│  │        if narrative: TextChunker   → 350-700 tokens, 75-token overlap     │  │
│  │        if table:     TableChunker  →                                     │  │
│  │                        table_schema chunk (title + headers)              │  │
│  │                        table_rows chunks (10-25 rows/chunk, col=val)     │  │
│  │                        table_summary chunk (optional, large tables)      │  │
│  │                                                                          │  │
│  │  each chunk → ChunkObject{                                               │  │
│  │    chunk_type, section_path[], display_text, search_text,                │  │
│  │    page_start, page_end, metadata{table_id?, bbox?, lang_hint, is_ocr}   │  │
│  │  }                                                                       │  │
│  │  writes: rag_chunks[] (status → CHUNKED)                                 │  │
│  └──────────────────────────────┬──────────────────────────────────────────┘  │
│                                 │                                              │
│  ┌──────────────────────────────▼──────────────────────────────────────────┐  │
│  │  Stage 4: EMBED  (embed_version = "bge-m3-v1")                          │  │
│  │                                                                          │  │
│  │  batch_embed(chunks[].search_text, model="BAAI/bge-m3")                  │  │
│  │    → vectors[]  (float32[1024])                                          │  │
│  │                                                                          │  │
│  │  writes: rag_chunks[].embedding  (status → EMBEDDED)                    │  │
│  └──────────────────────────────┬──────────────────────────────────────────┘  │
│                                 │                                              │
│  ┌──────────────────────────────▼──────────────────────────────────────────┐  │
│  │  Stage 5: INDEX                                                          │  │
│  │                                                                          │  │
│  │  UPDATE rag_chunks SET                                                   │  │
│  │    tsv = to_tsvector('malayalam', search_text)   -- FTS                  │  │
│  │    embedding = :vector                           -- pgvector             │  │
│  │  WHERE chunk_id = :id                                                    │  │
│  │                                                                          │  │
│  │  writes: rag_pdfs.status → INDEXED                                       │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  On any stage failure: rag_pdfs.status → ERROR, retry with backoff           │
│  After N failures: quarantine to rag_errors table                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Query Pipeline — Detailed Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PIPELINE                                    │
│                                                                               │
│  POST /search                                                                │
│  {                                                                           │
│    "query": "...",                                                           │
│    "filters": { "district": "...", "year_label": "...", ... },              │
│    "top_k": 10                                                               │
│  }                                                                           │
│                                                                               │
│  Step 1: Embed query                                                         │
│  ┌────────────────────────────────────────────────────────┐                 │
│  │  embed(query) → query_vector  (model: BAAI/bge-m3)     │                 │
│  └─────────────────────────┬──────────────────────────────┘                 │
│                            │                                                 │
│  Step 2: Parallel recall                                                     │
│  ┌─────────────────────────┴──────────────────────────────┐                 │
│  │  Dense recall (pgvector ANN)    │  Lexical recall (FTS) │                 │
│  │                                 │                        │                 │
│  │  SELECT chunk_id,               │  SELECT chunk_id,      │                 │
│  │    1 - (embedding <=> :vec)     │    ts_rank_cd(tsv,     │                 │
│  │    AS score                     │      query) AS score   │                 │
│  │  FROM rag_chunks                │  FROM rag_chunks,      │                 │
│  │  WHERE pdf_id IN (              │    plainto_tsquery(     │                 │
│  │    filter_subquery              │      :query) AS q      │                 │
│  │  )                              │  WHERE tsv @@ q        │                 │
│  │  ORDER BY embedding <=> :vec    │    AND filter...       │                 │
│  │  LIMIT 50                       │  ORDER BY score DESC   │                 │
│  │                                 │  LIMIT 50              │                 │
│  └─────────────┬───────────────────┴──────────┬─────────────┘                 │
│                │                              │                              │
│  Step 3: RRF fusion                                                          │
│  ┌─────────────┴──────────────────────────────┘                              │
│  │  rrf_score(chunk_id) =                                                   │
│  │    Σ  1 / (rrf_k + rank_i)   for each recall list i                      │
│  │                                                                          │
│  │  rrf_k = 60  (default)                                                   │
│  │  Sort by rrf_score DESC → top 40                                         │
│  └──────────────────────────────┬───────────────────────────────────────────┘│
│                                 │                                              │
│  Step 4: Rerank                                                              │
│  ┌──────────────────────────────▼───────────────────────────────────────────┐ │
│  │  CrossEncoderReranker(                                                   │ │
│  │    model="jinaai/jina-reranker-v2-base-multilingual",                    │ │
│  │    query=query,                                                          │ │
│  │    passages=[chunk.display_text for chunk in top40]                      │ │
│  │  ) → reranked_scores[]                                                   │ │
│  └──────────────────────────────┬───────────────────────────────────────────┘ │
│                                 │                                              │
│  Step 5: Assemble response                                                   │
│  ┌──────────────────────────────▼───────────────────────────────────────────┐ │
│  │  top_k chunks with:                                                      │ │
│  │    chunk_id, pdf_id, chunk_type                                          │ │
│  │    page_start, page_end                                                  │ │
│  │    display_text                                                          │ │
│  │    score (reranker score)                                                │ │
│  │    rag_metadata {district, local_body, year_label, lb_type,              │ │
│  │                  project_no, project_name, formulation, expense}         │ │
│  │    citation {gcs_path, gcs_bucket, page_start, page_end}                 │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Model (RAG Postgres)

### `rag_pdfs`

```sql
CREATE TABLE rag_pdfs (
    pdf_id          UUID        PRIMARY KEY,  -- from scraper pdfs.id
    project_id      UUID        NOT NULL,     -- from scraper projects.id (traceability)
    content_hash    TEXT        NOT NULL,     -- SHA256; used for idempotency
    gcs_bucket      TEXT        NOT NULL,
    gcs_path        TEXT        NOT NULL,
    downloaded_at   TIMESTAMPTZ,
    rag_metadata    JSONB       NOT NULL DEFAULT '{}',
    -- {district, local_body, year_label, lb_type, project_no, project_name,
    --  formulation, expense}
    parse_version   TEXT        NOT NULL DEFAULT '',
    chunk_version   TEXT        NOT NULL DEFAULT '',
    embed_version   TEXT        NOT NULL DEFAULT '',
    status          TEXT        NOT NULL DEFAULT 'NEW',
    -- NEW | DOWNLOADING | PARSED | CHUNKED | EMBEDDED | INDEXED | ERROR
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_rag_pdfs_status ON rag_pdfs (status);
CREATE INDEX idx_rag_pdfs_updated ON rag_pdfs (updated_at);
CREATE INDEX idx_rag_pdfs_meta ON rag_pdfs USING gin (rag_metadata jsonb_path_ops);
```

### `rag_chunks`

```sql
CREATE TABLE rag_chunks (
    chunk_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_id          UUID        NOT NULL REFERENCES rag_pdfs(pdf_id) ON DELETE CASCADE,
    page_start      INT         NOT NULL,
    page_end        INT         NOT NULL,
    chunk_type      TEXT        NOT NULL,
    -- kv | narrative | table_schema | table_rows | table_summary
    section_path    TEXT[]      NOT NULL DEFAULT '{}',
    display_text    TEXT        NOT NULL,   -- pretty; shown to users
    search_text     TEXT        NOT NULL,   -- normalised; used for FTS + embedding
    tsv             TSVECTOR    GENERATED ALWAYS AS (
                        to_tsvector('simple', search_text)
                    ) STORED,
    embedding       VECTOR(1024),           -- pgvector; NULL until Stage 4
    metadata        JSONB       NOT NULL DEFAULT '{}',
    -- {table_id?, bbox?, lang_hint, is_ocr, token_count}
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_rag_chunks_pdf    ON rag_chunks (pdf_id, page_start);
CREATE INDEX idx_rag_chunks_tsv    ON rag_chunks USING gin (tsv);
CREATE INDEX idx_rag_chunks_emb    ON rag_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_rag_chunks_type   ON rag_chunks (chunk_type);
CREATE INDEX idx_rag_chunks_meta   ON rag_chunks USING gin (metadata jsonb_path_ops);
```

### `rag_blocks` (optional; for debugging/traceability)

```sql
CREATE TABLE rag_blocks (
    block_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_id          UUID        NOT NULL REFERENCES rag_pdfs(pdf_id) ON DELETE CASCADE,
    page_index      INT         NOT NULL,
    block_type      TEXT        NOT NULL,   -- text | table | figure | header | footer
    raw_text        TEXT,
    bbox            JSONB,                  -- {x0, y0, x1, y1}
    reading_order   INT,
    is_ocr          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### `rag_tables` (optional; for full table traceability)

```sql
CREATE TABLE rag_tables (
    table_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_id          UUID        NOT NULL REFERENCES rag_pdfs(pdf_id) ON DELETE CASCADE,
    page_index      INT         NOT NULL,
    caption         TEXT,
    headers         JSONB,      -- [col_name, ...]
    rows_json       JSONB,      -- [[val, ...], ...]
    html            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### `rag_ingestion_runs`

```sql
CREATE TABLE rag_ingestion_runs (
    run_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_id          UUID        NOT NULL REFERENCES rag_pdfs(pdf_id),
    parse_version   TEXT        NOT NULL,
    chunk_version   TEXT        NOT NULL,
    embed_version   TEXT        NOT NULL,
    status          TEXT        NOT NULL,   -- STARTED | SUCCESS | FAILED
    stage_reached   TEXT,                   -- DOWNLOAD | PARSE | CHUNK | EMBED | INDEX
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    error           TEXT,
    metrics         JSONB       NOT NULL DEFAULT '{}'
    -- {parse_ms, chunk_ms, embed_ms, index_ms, token_counts, chunk_count}
);
```

### `rag_errors` (poison pill quarantine)

```sql
CREATE TABLE rag_errors (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_id          UUID        NOT NULL,
    stage           TEXT        NOT NULL,
    error           TEXT        NOT NULL,
    attempt_count   INT         NOT NULL DEFAULT 1,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 7. Parsing and Document Intelligence

### Parser choice

| Parser | Role | Notes |
|---|---|---|
| **Marker** | Primary; born-digital path | Converts PDF → markdown/JSON/blocks; handles multi-column, tables, forms; removes headers/footers; all languages including Malayalam; CPU/GPU/MPS |
| **Surya** | OCR fallback + table extraction | Layout analysis, reading order, table recognition → JSON/HTML output |

### Parse decision policy

```
is_born_digital = pdf_has_extractable_text(pdf_bytes)
quality_score   = assess_extraction_quality(marker_output)

if is_born_digital and quality_score >= QUALITY_THRESHOLD:
    use Marker output
elif is_born_digital and quality_score < QUALITY_THRESHOLD:
    run Surya OCR + table extraction; merge with Marker blocks
else:  # scanned / image-only PDF
    run Surya full OCR pipeline
    flag block.is_ocr = True
```

`QUALITY_THRESHOLD`: configurable (default 0.6); computed as mean of: text coverage ratio, character confidence, block coherence score.

### `ParseResult` schema

```python
@dataclass
class ParseResult:
    pdf_id: str
    page_count: int
    is_ocr: bool
    quality_score: float
    parse_version: str
    blocks: list[Block]   # reading-order sorted, across all pages
    tables: list[Table]
    pages: list[Page]     # per-page metadata (bbox, rotation, etc.)
```

---

## 8. Chunking Specification

### Token budget defaults

| Parameter | Default |
|---|---|
| KV chunk target | 400–600 tokens |
| KV fields per chunk | 1–5 contiguous fields |
| Narrative chunk target | 350–700 tokens |
| Narrative overlap | 75 tokens |
| Table rows per chunk | 10–25 (width-adjusted) |
| Max chunk size (hard limit) | 800 tokens |

### Chunk types

#### 1) KV chunks — form fields

Detect label+value pairs in blocks. Bundle 1–5 contiguous fields.

```
[section_path]
Field label 1: value 1
Field label 2: value 2
...
```

`display_text`: formatted as above with section header.  
`search_text`: `{section}. {label_1} {value_1}. {label_2} {value_2}.` (normalised, punctuation stripped, Unicode-normalised NFC for Malayalam).

#### 2) Narrative chunks — paragraphs/sections

Group paragraphs under a section heading. Sliding window with overlap.

```
[section_path]
<paragraph text>
```

#### 3) Table chunks — first-class table handling

For each `Table` in parse output:

**3a) `table_schema` chunk** (always generated):
```
Table: {caption or inferred title}
Section: {section_path}
Columns: {col1} | {col2} | {col3} ...
Units: {unit hints from header if present}
{1 row of sample data}
```

**3b) `table_rows` chunks** (one or more):
```
Table: {caption}  [rows {start}–{end}]
{col1}={val1}, {col2}={val2}, {col3}={val3}
{col1}={val1}, {col2}={val2}, {col3}={val3}
...
```
Headers repeated at top of each chunk for standalone retrieval.

**3c) `table_summary` chunk** (optional; for tables > 50 rows):
```
This table lists {inferred_topic} for {section_context}.
Key columns: {col list}. There are {N} rows representing {inferred_row_entity}.
```

### `ChunkObject` schema

```python
@dataclass
class ChunkObject:
    chunk_type: str          # kv | narrative | table_schema | table_rows | table_summary
    section_path: list[str]  # e.g. ["Section A", "Sub-section 2"]
    display_text: str        # pretty text shown to users
    search_text: str         # normalised retrieval text
    page_start: int
    page_end: int
    metadata: dict           # table_id?, bbox?, lang_hint, is_ocr, token_count
```

---

## 9. Embeddings and Reranking

### Embedding models

| Model | Dimensions | Context | Notes |
|---|---|---|---|
| **BGE-M3** (`BAAI/bge-m3`) | 1024 | 8192 tokens | **Default v1**; multi-lingual, multi-function (dense + sparse + ColBERT); strong Malayalam |
| Qwen3-Embedding | 4096 | 32768 | Challenger; benchmark vs BGE-M3 on eval set |

Both are OSS, available on HuggingFace, can run on CPU or GPU.

**v1 default:** BGE-M3 (dimension: 1024). Vector column: `VECTOR(1024)`.

### Reranker

**`jinaai/jina-reranker-v2-base-multilingual`** — multilingual cross-encoder; OSS on HuggingFace.

Input: `(query, passage)` pairs.  
Output: scalar relevance scores.

Applied to top-40 RRF-fused candidates, returning top `top_k` (default 10).

### Batch inference

- Embedding: batch size 64 (GPU) / 16 (CPU); max token length 512 for retrieval use case (BGE-M3 docs recommend ≤512 for retrieval despite 8192 support).
- Reranking: batch size 32.
- Models loaded once at worker startup; cached in memory.

---

## 10. Hybrid Retrieval and Fusion

### Recall stage

```sql
-- Dense recall
SELECT c.chunk_id,
       1 - (c.embedding <=> $1::vector) AS score,
       ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector) AS rank
FROM rag_chunks c
JOIN rag_pdfs p ON p.pdf_id = c.pdf_id
WHERE ($2::text IS NULL OR p.rag_metadata->>'district'   = $2)
  AND ($3::text IS NULL OR p.rag_metadata->>'year_label' = $3)
  AND ($4::text IS NULL OR p.rag_metadata->>'local_body' = $4)
  AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> $1::vector
LIMIT 50;

-- Lexical recall
SELECT c.chunk_id,
       ts_rank_cd(c.tsv, query) AS score,
       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, query) DESC) AS rank
FROM rag_chunks c
JOIN rag_pdfs p ON p.pdf_id = c.pdf_id,
     plainto_tsquery('simple', $1) AS query
WHERE c.tsv @@ query
  AND ($2::text IS NULL OR p.rag_metadata->>'district'   = $2)
  AND ($3::text IS NULL OR p.rag_metadata->>'year_label' = $3)
  AND ($4::text IS NULL OR p.rag_metadata->>'local_body' = $4)
ORDER BY ts_rank_cd(c.tsv, query) DESC
LIMIT 50;
```

### RRF fusion

```python
RRF_K = 60  # default; tune via eval harness

def rrf_fuse(dense_ranked: list[RankedChunk],
             fts_ranked: list[RankedChunk]) -> list[RankedChunk]:
    scores: dict[str, float] = defaultdict(float)
    for i, chunk in enumerate(dense_ranked):
        scores[chunk.chunk_id] += 1.0 / (RRF_K + i + 1)
    for i, chunk in enumerate(fts_ranked):
        scores[chunk.chunk_id] += 1.0 / (RRF_K + i + 1)
    return sorted(all_chunks, key=lambda c: scores[c.chunk_id], reverse=True)[:40]
```

### Rerank

```python
reranker = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual")

def rerank(query: str, candidates: list[RankedChunk], top_k: int) -> list[RankedChunk]:
    pairs = [(query, c.display_text) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
```

---

## 11. Service API Contracts

### Ingestion API

**`POST /ingest/pdf`**

```json
// Request
{ "pdf_id": "uuid" }

// Response 202
{
  "run_id": "uuid",
  "pdf_id": "uuid",
  "status": "QUEUED"
}
```

**`GET /ingest/status?pdf_id={uuid}`**

```json
{
  "pdf_id": "uuid",
  "status": "INDEXED",
  "stage_reached": "INDEX",
  "parse_version": "marker-v1.0",
  "chunk_version": "v1.0",
  "embed_version": "bge-m3-v1",
  "chunk_count": 42,
  "run_id": "uuid",
  "updated_at": "2026-03-12T10:00:00Z",
  "error": null
}
```

**`POST /ingest/backfill`**

```json
// Request — trigger batch reprocessing
{
  "filters": { "district": "Ernakulam" },
  "target_version": {
    "embed_version": "qwen3-v1"
  }
}

// Response 202
{ "job_id": "uuid", "matched_pdfs": 1240 }
```

### Query API

**`POST /search`**

```json
// Request
{
  "query": "formulation amount for road widening in Chengamanad",
  "filters": {
    "district": "Ernakulam",
    "local_body": "Chengamanad Grama Panchayat",
    "year_label": "2024-2025",
    "lb_type": "Grama Panchayat"
  },
  "top_k": 10
}

// Response 200
{
  "query": "...",
  "results": [
    {
      "chunk_id": "uuid",
      "pdf_id": "uuid",
      "chunk_type": "kv",
      "page_start": 3,
      "page_end": 3,
      "display_text": "Section A\nFormulation Amount: ₹12,50,000\nExpense Type: Capital",
      "score": 0.94,
      "rag_metadata": {
        "district": "Ernakulam",
        "local_body": "Chengamanad Grama Panchayat",
        "year_label": "2024-2025",
        "lb_type": "Grama Panchayat",
        "project_no": "EP-24-001",
        "project_name": "Road widening - NH bypass link",
        "formulation": "12,50,000",
        "expense": "Capital"
      },
      "citation": {
        "gcs_bucket": "sulekha-pdfs",
        "gcs_path": "pdfs/2024-2025/Grama_Panchayat/Ernakulam/...",
        "page_start": 3,
        "page_end": 3
      }
    }
  ],
  "latency_ms": {
    "embed": 42,
    "recall": 38,
    "rerank": 110,
    "total": 195
  }
}
```

---

## 12. Directory Structure

```
sulekha-preprocessing/
│
├── docs/
│   └── SPEC.md                      ← this file
│
├── alembic/                         ← RAG DB migrations
│   ├── env.py
│   ├── alembic.ini
│   └── versions/
│       ├── 0001_rag_initial_schema.py
│       └── ...
│
├── src/
│   └── sambandh/
│       ├── __init__.py
│       │
│       ├── config.py                ← Settings (pydantic-settings)
│       │   # SCRAPER_DATABASE_URL, RAG_DATABASE_URL, GCS_*, model names, versions
│       │
│       ├── db/
│       │   ├── __init__.py
│       │   ├── models.py            ← SQLAlchemy models (rag_pdfs, rag_chunks, ...)
│       │   └── repositories.py     ← DB query helpers
│       │
│       ├── scraper_db/
│       │   ├── __init__.py
│       │   └── reader.py           ← Read-only access to scraper Postgres
│       │   # fetch_uploaded_pdfs_since(checkpoint)
│       │   # build_rag_metadata(pdf_id) → rag_metadata dict
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   └── gcs.py              ← GCS download (reused pattern from scraper)
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── poller.py           ← Periodic poll → enqueue jobs
│       │   ├── pipeline.py         ← Orchestrates stages 1–5
│       │   ├── stages/
│       │   │   ├── __init__.py
│       │   │   ├── download.py     ← Stage 1: GCS download + hash verify
│       │   │   ├── parse.py        ← Stage 2: Marker + Surya
│       │   │   ├── chunk.py        ← Stage 3: KV + narrative + table chunker
│       │   │   ├── embed.py        ← Stage 4: BGE-M3 batch embedding
│       │   │   └── index.py        ← Stage 5: FTS + pgvector upsert
│       │   └── worker.py           ← Celery worker (or ARQ / plain asyncio)
│       │
│       ├── parsing/
│       │   ├── __init__.py
│       │   ├── marker_parser.py    ← Marker integration + ParseResult builder
│       │   ├── surya_parser.py     ← Surya OCR + table extraction
│       │   ├── quality.py          ← Quality scoring / OCR fallback decision
│       │   └── models.py           ← ParseResult, Block, Table, Page dataclasses
│       │
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base.py             ← ChunkObject + base chunker interface
│       │   ├── kv_chunker.py       ← Form field chunker
│       │   ├── text_chunker.py     ← Narrative sliding-window chunker
│       │   ├── table_chunker.py    ← Table schema/rows/summary chunker
│       │   └── dispatcher.py      ← Block → chunker routing
│       │
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── base.py             ← EmbedderProtocol interface
│       │   ├── bge_m3.py           ← BGE-M3 implementation
│       │   ├── qwen3.py            ← Qwen3-Embedding (challenger)
│       │   └── reranker.py         ← Jina reranker cross-encoder
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── dense.py            ← pgvector ANN query
│       │   ├── fts.py              ← Postgres FTS query
│       │   ├── fusion.py           ← RRF implementation
│       │   └── pipeline.py         ← Orchestrates dense+fts+rrf+rerank
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py             ← FastAPI app
│       │   ├── routes/
│       │   │   ├── ingest.py       ← POST /ingest/pdf, GET /ingest/status
│       │   │   └── search.py       ← POST /search
│       │   └── schemas.py          ← Pydantic request/response models
│       │
│       └── cli.py                  ← CLI (typer/click): ingest, search, status, backfill
│
├── eval/
│   ├── README.md
│   ├── queries/
│   │   ├── malayalam.jsonl         ← Malayalam-only queries + expected chunk refs
│   │   ├── english.jsonl
│   │   ├── mixed.jsonl
│   │   └── table_lookups.jsonl
│   └── harness.py                  ← Recall@50, MRR@10, nDCG@10, table hit rate
│
├── scripts/
│   ├── backfill_embeddings.py      ← Re-embed all chunks for version upgrade
│   └── sample_queries.py           ← Ad-hoc search testing
│
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_parsing.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── docker-compose.yml               ← RAG stack (rag-postgres, redis, ingestion-worker, api)
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 13. Infrastructure and Deployment

### Network assumption

The `sulekha-preprocessing` stack connects to the **existing Docker network** `sulekha-network` (created by the scraper stack). No shared compose files are needed. The scraper stack must be started first; its Postgres (`sulekha-postgres`) and GCS bucket are accessed read-only.

### `docker-compose.yml` (this repo)

```yaml
# sulekha-preprocessing/docker-compose.yml

networks:
  default:
    name: sulekha-network
    external: true   # joins scraper's network; scraper-postgres accessible by hostname

services:

  # ── RAG Database ────────────────────────────────────────────────────────────
  rag-postgres:
    image: pgvector/pgvector:pg16
    container_name: sambandh-postgres
    environment:
      POSTGRES_USER: sambandh
      POSTGRES_PASSWORD: sambandh
      POSTGRES_DB: sambandh
    volumes:
      - rag_postgres_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"     # different host port to avoid collision with scraper postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sambandh"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # ── Job Queue ────────────────────────────────────────────────────────────────
  rag-redis:
    image: redis:7-alpine
    container_name: sambandh-redis
    ports:
      - "6380:6379"     # different host port
    restart: unless-stopped

  # ── DB Migration (one-off) ───────────────────────────────────────────────────
  migrate:
    build: .
    container_name: sambandh-migrate
    command: alembic upgrade head
    environment:
      - RAG_DATABASE_URL=postgresql+psycopg://sambandh:sambandh@rag-postgres:5432/sambandh
      - SCRAPER_DATABASE_URL=postgresql+psycopg://sulekha:sulekha@sulekha-postgres:5432/sulekha
    depends_on:
      rag-postgres:
        condition: service_healthy
    profiles:
      - migrate

  # ── Ingestion Worker ─────────────────────────────────────────────────────────
  ingestion-worker:
    build: .
    container_name: sambandh-ingestion
    command: sambandh ingest worker
    environment:
      - RAG_DATABASE_URL=postgresql+psycopg://sambandh:sambandh@rag-postgres:5432/sambandh
      - SCRAPER_DATABASE_URL=postgresql+psycopg://sulekha:sulekha@sulekha-postgres:5432/sulekha
      - RAG_REDIS_URL=redis://rag-redis:6379/0
      - STORAGE_BACKEND=gcs
      - GCS_BUCKET_NAME=sulekha-pdfs
      - GCS_PROJECT_ID=sulekha-487304
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcs-key.json
      - EMBED_MODEL=BAAI/bge-m3
      - RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
      - PARSE_VERSION=marker-v1.0
      - CHUNK_VERSION=v1.0
      - EMBED_VERSION=bge-m3-v1
      - LOG_LEVEL=INFO
    volumes:
      - ./credentials/gcs-key.json:/app/credentials/gcs-key.json:ro
      - model_cache:/app/.cache/huggingface
    depends_on:
      rag-postgres:
        condition: service_healthy
      rag-redis:
        condition: service_healthy
    restart: unless-stopped

  # ── Query API ────────────────────────────────────────────────────────────────
  query-api:
    build: .
    container_name: sambandh-api
    command: sambandh api serve --host 0.0.0.0 --port 8000
    environment:
      - RAG_DATABASE_URL=postgresql+psycopg://sambandh:sambandh@rag-postgres:5432/sambandh
      - EMBED_MODEL=BAAI/bge-m3
      - RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
      - LOG_LEVEL=INFO
    volumes:
      - model_cache:/app/.cache/huggingface
    ports:
      - "8000:8000"
    depends_on:
      rag-postgres:
        condition: service_healthy
    restart: unless-stopped

volumes:
  rag_postgres_data:
  model_cache:
```

### Starting both stacks locally

```bash
# 1. Start scraper stack (creates sulekha-network)
cd ~/sulekha-scraper
docker compose -f docker-compose.base.yml up -d

# 2. Start RAG stack (joins sulekha-network; sees sulekha-postgres)
cd ~/sulekha-preprocessing
docker compose --profile migrate up migrate   # run migrations once
docker compose up -d ingestion-worker query-api
```

### Environment config (`.env.example`)

```env
# RAG database
RAG_DATABASE_URL=postgresql+psycopg://sambandh:sambandh@rag-postgres:5432/sambandh

# Scraper database (read-only)
SCRAPER_DATABASE_URL=postgresql+psycopg://sulekha:sulekha@sulekha-postgres:5432/sulekha

# Job queue
RAG_REDIS_URL=redis://rag-redis:6379/0

# GCS (PDF download)
STORAGE_BACKEND=gcs
GCS_BUCKET_NAME=sulekha-pdfs
GCS_PROJECT_ID=sulekha-487304
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcs-key.json

# Models
EMBED_MODEL=BAAI/bge-m3
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual

# Versioning (bump to trigger reprocessing)
PARSE_VERSION=marker-v1.0
CHUNK_VERSION=v1.0
EMBED_VERSION=bge-m3-v1

# Ingestion poller
INGESTION_POLL_INTERVAL_SECONDS=300

# Retrieval
RRF_K=60
DENSE_RECALL_K=50
FTS_RECALL_K=50
RERANK_CANDIDATE_K=40

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 14. Idempotency and Versioning

### Processing key

```
(pdf_id, content_hash, parse_version, chunk_version, embed_version)
```

### Decision matrix

| Condition | Action |
|---|---|
| Key unchanged, `rag_pdfs.status = INDEXED` | Skip entirely |
| `content_hash` changed (PDF updated) | Full re-run: re-parse + re-chunk + re-embed + re-index |
| Only `embed_version` changed | Delete + re-embed existing chunks; re-index embeddings only |
| `parse_version` or `chunk_version` changed | Delete existing chunks + re-parse + re-chunk + re-embed + re-index |
| Status is ERROR, versions unchanged | Retry (with backoff); quarantine after N=3 failures |

### Backfill for model upgrades

```bash
# Bump EMBED_VERSION in .env, then:
sambandh ingest backfill --embed-version qwen3-v1 [--filter district=Ernakulam]
```

Backfill re-embeds chunks in batches of 500 PDFs, updating `rag_chunks.embedding` in place, then refreshes HNSW index.

---

## 15. Observability

### Metrics to track (via structlog + optional Prometheus)

| Metric | Type | Notes |
|---|---|---|
| `ingestion_pdfs_total{status}` | counter | NEW, INDEXED, ERROR |
| `ingestion_stage_duration_seconds{stage}` | histogram | download, parse, chunk, embed, index |
| `ingestion_parse_ocr_fallback_total` | counter | OCR path activations |
| `ingestion_chunk_count{type}` | histogram | kv, narrative, table_schema, table_rows |
| `query_duration_seconds{stage}` | histogram | embed, recall, rerank, total |
| `query_recall_mix{source}` | histogram | % dense-only, fts-only, both |
| `embedding_batch_size` | histogram | actual batch sizes |

### Structured log fields (per stage)

```json
{
  "event": "stage_complete",
  "pdf_id": "uuid",
  "run_id": "uuid",
  "stage": "PARSE",
  "parse_version": "marker-v1.0",
  "is_ocr": false,
  "quality_score": 0.84,
  "duration_ms": 1420,
  "block_count": 38,
  "table_count": 5
}
```

---

## 16. Evaluation Plan

### Eval dataset

50–200 queries covering:

| Category | Example queries |
|---|---|
| Malayalam-only | `ഫോർമുലേഷൻ തുക?`, `ഗുണഭോക്തൃ വിഭാഗം?` |
| English-only | `What is the formulation amount for road project?` |
| Mixed / code-switch | `project name and formulation amount` |
| Table cell lookup | `allocation for drinking water in 2024-2025` |
| Entity lookup | `project EP-24-001 expense type` |

Each query: 1–5 labeled relevant chunk IDs (or at minimum: page references).

### Metrics

| Metric | Target |
|---|---|
| Recall@50 (hybrid) | ≥ 0.90 |
| MRR@10 (after rerank) | ≥ 0.70 |
| nDCG@10 (after rerank) | ≥ 0.75 |
| Table hit rate (table chunks in top 10) | ≥ 0.80 |
| p95 query latency | < 1.5s |

### Benchmark runs

1. **Chunking:** baseline (narrative only) vs. +table rows vs. +KV chunks
2. **Embedding:** BGE-M3 vs. Qwen3-Embedding
3. **Fusion:** dense-only vs. fts-only vs. hybrid+RRF
4. **Reranking:** with vs. without jina-reranker

---

## 17. Milestones

### V0 — Skeleton (1–2 weeks)

- [ ] RAG DB schema + migrations
- [ ] Scraper DB reader: `fetch_uploaded_pdfs_since()` + `build_rag_metadata()`
- [ ] GCS download stage
- [ ] Marker parse stage (no OCR fallback yet)
- [ ] Basic text chunker (narrative only)
- [ ] Store chunks without embeddings
- [ ] Status API (`GET /ingest/status`)
- [ ] Filter-only query (no vector; by metadata only)

### V1 — Full retrieval (2–4 weeks)

- [ ] KV chunker + table chunker
- [ ] BGE-M3 embedding + pgvector HNSW indexing
- [ ] FTS indexing + hybrid recall + RRF
- [ ] Jina reranker integration
- [ ] Full search API (`POST /search`) with citations
- [ ] Evaluation harness (50 queries)

### V1.5 — Production hardening (4–6 weeks)

- [ ] Surya OCR fallback + quality scoring
- [ ] Versioned backfill tooling
- [ ] Structured metrics + Prometheus/Grafana setup
- [ ] Sakarma PDFs as second source (`source_type` field in schema)
- [ ] Challenger embedding benchmark (Qwen3-Embedding)
- [ ] Production evaluation (200 queries, full report)

---

## 18. Open Questions

| # | Question | Status |
|---|---|---|
| 1 | Include Sakarma meeting PDFs in v1 or defer to v1.5? | **Deferred to v1.5** |
| 2 | Read-only Postgres user for scraper DB? | Nice-to-have; document in scraper README |
| 3 | GPU availability for embedding/reranker? | CPU works; GPU recommended for throughput |
| 4 | FTS language config: `simple` vs. custom Malayalam dictionary? | Start with `simple`; custom in v1.5 |
| 5 | HNSW index params (m, ef_construction)? | Default (m=16, ef=64); tune post-eval |
| 6 | Reranker: Jina v2 vs. Qwen3-Reranker? | Benchmark in v1.5 |
