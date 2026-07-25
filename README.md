# Sambandh — Preprocessing & RAG Pipeline

The document-ingestion and retrieval pipeline behind [GramSAMBANDH](https://github.com/tushar-anand15/sambandh-app) — the system for searching Kerala municipal & panchayat project records.

This is where messy public PDFs become a queryable, multilingual index. It takes scanned/native local-government project documents (Malayalam + English) and produces a hybrid search index with page-level citations.

## Pipeline

1. **Ingest** — pull source PDFs (project documents from Kerala local bodies).
2. **Parse** — layout-preserving extraction, including tables and form fields, with OCR for scanned pages.
3. **Chunk** — structure-aware chunking tuned for multilingual retrieval.
4. **Embed & index** — dense embeddings into `pgvector` alongside Postgres full-text search (lexical), for hybrid recall.
5. **Retrieve** — hybrid recall → RRF fusion → reranking, returning ranked chunks with project metadata and page citations.

See `docs/SPEC.md` for the full technical specification, `docs/OCR_BENCHMARK.md` for parsing/OCR benchmarks, and `docs/GPU_SETUP.md` for GPU notes.

## Running

```bash
uv sync
cp etc/*.example .            # configure sources + DB (see docs/SPEC.md)
# run the ingestion pipeline
uv run python -m src ...      # see docs/SPEC.md for pipeline entry points
```

Docker:

```bash
docker compose up
```

## Why

Local-government spending data in India is public in principle but locked inside thousands of inconsistent PDFs. This pipeline is the reusable machinery for turning that kind of source into something you can actually query.
