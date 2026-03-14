#!/usr/bin/env bash
# Sync RAG data (documents + chunks + embeddings) from the preprocessing
# pipeline DB into the standalone app DB.
#
# Two-step workflow:
#
#   1. DUMP (pipeline side) — export tables to a directory
#      PIPELINE_DB_URL="postgresql://sambandh:sambandh@localhost:5433/sambandh" \
#        bash scripts/sync_to_app.sh dump ./rag_export
#
#   2. LOAD (app side) — restore the dump into the app DB
#      APP_DB_URL="postgresql://sambandh:sambandh@localhost:5434/sambandh" \
#        bash scripts/sync_to_app.sh load ./rag_export
#
# Load is incremental (upsert) — re-running skips rows that already
# exist. Use --fresh to truncate and reload from scratch instead.
#
# The dump directory can be copied between machines (scp, GCS, etc.)
# before running the load step.

set -euo pipefail

# Documents: PDF-level metadata extracted from parse_jobs.
DOC_COLS="pdf_id, district_name, lb_name, lb_type, project_no, project_name, year_label, gcs_path, gcs_bucket, file_size_bytes, page_count, block_count, table_count, created_at"

# Chunks: app schema drops chunk_job_id (pipeline-only FK) and tsv (generated).
PC_COLS="chunk_id, pdf_id, chunk_index, chunk_type, section_path, display_text, search_text, page_start, page_end, token_count, metadata, district_name, lb_name, lb_type, project_no, year_label, created_at"

# Embeddings: synced as-is.
CE_COLS="id, chunk_id, embed_model, embedding, created_at"

usage() {
  echo "Usage:"
  echo "  $0 dump  <output_dir>             Export from PIPELINE_DB_URL"
  echo "  $0 load  <input_dir>  [--fresh]   Load into APP_DB_URL (incremental by default)"
  exit 1
}

cmd_dump() {
  local dir="$1"
  local db="${PIPELINE_DB_URL:?Set PIPELINE_DB_URL}"

  mkdir -p "$dir"

  echo "==> Dumping documents (from parse_jobs where status=done)..."
  psql "$db" -c "\COPY (SELECT $DOC_COLS FROM parse_jobs WHERE status = 'done' ORDER BY district_name, lb_name, project_no) TO STDOUT" \
    > "$dir/documents.dump"

  echo "==> Dumping processed_chunks..."
  psql "$db" -c "\COPY (SELECT $PC_COLS FROM processed_chunks ORDER BY pdf_id, chunk_index) TO STDOUT" \
    > "$dir/processed_chunks.dump"

  echo "==> Dumping chunk_embeddings..."
  psql "$db" -c "\COPY (SELECT $CE_COLS FROM chunk_embeddings) TO STDOUT" \
    > "$dir/chunk_embeddings.dump"

  wc -l < "$dir/documents.dump"         > "$dir/.docs_count"
  wc -l < "$dir/processed_chunks.dump"  > "$dir/.chunks_count"
  wc -l < "$dir/chunk_embeddings.dump"  > "$dir/.embeddings_count"

  echo ""
  echo "    documents:  $(cat "$dir/.docs_count")"
  echo "    chunks:     $(cat "$dir/.chunks_count")"
  echo "    embeddings: $(cat "$dir/.embeddings_count")"
  echo "    size:       $(du -sh "$dir" | cut -f1)"
  echo ""
  echo "==> Dump written to $dir"
}

cmd_load() {
  local dir="$1"
  local fresh="${2:-}"
  local db="${APP_DB_URL:?Set APP_DB_URL}"

  for f in documents.dump processed_chunks.dump chunk_embeddings.dump; do
    if [ ! -f "$dir/$f" ]; then
      echo "ERROR: $dir/$f not found"
      exit 1
    fi
  done

  echo "==> Loading into app DB..."
  echo "    Source: $dir"
  echo "    documents:  $(cat "$dir/.docs_count" 2>/dev/null || wc -l < "$dir/documents.dump")"
  echo "    chunks:     $(cat "$dir/.chunks_count" 2>/dev/null || wc -l < "$dir/processed_chunks.dump")"
  echo "    embeddings: $(cat "$dir/.embeddings_count" 2>/dev/null || wc -l < "$dir/chunk_embeddings.dump")"
  echo "    mode:       $([ "$fresh" = "--fresh" ] && echo "fresh (truncate + reload)" || echo "incremental (upsert, skip existing)")"
  echo ""

  if [ "$fresh" = "--fresh" ]; then
    echo "    Truncating all tables..."
    psql "$db" -c "TRUNCATE chunk_embeddings, processed_chunks, documents CASCADE;"

    echo "    Loading documents..."
    psql "$db" -c "\COPY documents ($DOC_COLS) FROM STDIN" < "$dir/documents.dump"

    echo "    Loading processed_chunks..."
    psql "$db" -c "\COPY processed_chunks ($PC_COLS) FROM STDIN" < "$dir/processed_chunks.dump"

    echo "    Loading chunk_embeddings..."
    psql "$db" -c "\COPY chunk_embeddings ($CE_COLS) FROM STDIN" < "$dir/chunk_embeddings.dump"
  else
    # Incremental: load into temp tables, then INSERT ... ON CONFLICT DO NOTHING.
    # Temp tables live for the psql session — no ON COMMIT DROP needed.
    echo "    Upserting documents..."
    psql "$db" <<SQL
CREATE TEMP TABLE _tmp_docs (LIKE documents INCLUDING DEFAULTS);
\COPY _tmp_docs ($DOC_COLS) FROM '$dir/documents.dump'
INSERT INTO documents ($DOC_COLS)
  SELECT $DOC_COLS FROM _tmp_docs
  ON CONFLICT (pdf_id) DO NOTHING;
DROP TABLE _tmp_docs;
SQL

    echo "    Upserting processed_chunks..."
    psql "$db" <<SQL
CREATE TEMP TABLE _tmp_chunks (LIKE processed_chunks INCLUDING DEFAULTS);
\COPY _tmp_chunks ($PC_COLS) FROM '$dir/processed_chunks.dump'
INSERT INTO processed_chunks ($PC_COLS)
  SELECT $PC_COLS FROM _tmp_chunks
  ON CONFLICT (chunk_id) DO NOTHING;
DROP TABLE _tmp_chunks;
SQL

    echo "    Upserting chunk_embeddings..."
    psql "$db" <<SQL
CREATE TEMP TABLE _tmp_embeds (LIKE chunk_embeddings INCLUDING DEFAULTS);
\COPY _tmp_embeds ($CE_COLS) FROM '$dir/chunk_embeddings.dump'
INSERT INTO chunk_embeddings ($CE_COLS)
  SELECT $CE_COLS FROM _tmp_embeds
  ON CONFLICT (chunk_id, embed_model) DO NOTHING;
DROP TABLE _tmp_embeds;
SQL
  fi

  echo ""
  echo "==> Verifying..."
  psql "$db" --no-align --tuples-only -c "
    SELECT 'documents:        ' || count(*) FROM documents
    UNION ALL
    SELECT 'processed_chunks: ' || count(*) FROM processed_chunks
    UNION ALL
    SELECT 'chunk_embeddings: ' || count(*) FROM chunk_embeddings;
  "
  echo ""
  echo "==> Load complete."
}

# ── Main ────────────────────────────────────────────────────────────

[ $# -ge 2 ] || usage

case "$1" in
  dump) cmd_dump "$2" ;;
  load) cmd_load "$2" "${3:-}" ;;
  *)    usage ;;
esac
