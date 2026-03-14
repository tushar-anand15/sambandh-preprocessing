"""Tier 2: Integration tests.

These require docker-compose up (Postgres + Redis) and are intended to be
run manually during the local integration testing phase.

Usage:
    docker compose up -d postgres redis
    DATABASE_URL=postgresql://sambandh:sambandh@localhost:5433/sambandh \
    REDIS_URL=redis://localhost:6380/0 \
    pytest tests/test_integration.py -v
"""

import os
import uuid

import psycopg
import pytest

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://sambandh:sambandh@localhost:5433/sambandh"
)


def _clean_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


@pytest.fixture
def db_conn():
    conn = psycopg.connect(_clean_url(DATABASE_URL), row_factory=psycopg.rows.dict_row)
    yield conn
    conn.rollback()
    conn.close()


class TestIdempotentEnqueue:
    def test_insert_new_job(self, db_conn):
        pdf_id = uuid.uuid4()
        row = db_conn.execute(
            """INSERT INTO parse_jobs (pdf_id, gcs_path, gcs_bucket, district_name)
               VALUES (%s, 'test/path.pdf', 'test-bucket', 'TestDistrict')
               ON CONFLICT (pdf_id) DO NOTHING
               RETURNING id""",
            [str(pdf_id)],
        ).fetchone()
        assert row is not None
        assert row["id"] is not None

    def test_duplicate_insert_returns_nothing(self, db_conn):
        pdf_id = uuid.uuid4()
        db_conn.execute(
            """INSERT INTO parse_jobs (pdf_id, gcs_path, gcs_bucket)
               VALUES (%s, 'test/a.pdf', 'bucket')""",
            [str(pdf_id)],
        )
        row = db_conn.execute(
            """INSERT INTO parse_jobs (pdf_id, gcs_path, gcs_bucket)
               VALUES (%s, 'test/a.pdf', 'bucket')
               ON CONFLICT (pdf_id) DO NOTHING
               RETURNING id""",
            [str(pdf_id)],
        ).fetchone()
        assert row is None

    def test_status_defaults_to_pending(self, db_conn):
        pdf_id = uuid.uuid4()
        db_conn.execute(
            """INSERT INTO parse_jobs (pdf_id, gcs_path, gcs_bucket)
               VALUES (%s, 'test/b.pdf', 'bucket')""",
            [str(pdf_id)],
        )
        row = db_conn.execute(
            "SELECT status FROM parse_jobs WHERE pdf_id = %s", [str(pdf_id)]
        ).fetchone()
        assert row["status"] == "pending"


class TestStatusTransitions:
    def _create_job(self, db_conn) -> str:
        pdf_id = uuid.uuid4()
        row = db_conn.execute(
            """INSERT INTO parse_jobs (pdf_id, gcs_path, gcs_bucket)
               VALUES (%s, 'test/c.pdf', 'bucket')
               RETURNING id""",
            [str(pdf_id)],
        ).fetchone()
        return str(row["id"])

    def test_pending_to_started(self, db_conn):
        job_id = self._create_job(db_conn)
        db_conn.execute(
            """UPDATE parse_jobs SET status = 'started', worker_id = 'w1'
               WHERE id = %s AND status = 'pending'""",
            [job_id],
        )
        row = db_conn.execute(
            "SELECT status, worker_id FROM parse_jobs WHERE id = %s", [job_id]
        ).fetchone()
        assert row["status"] == "started"
        assert row["worker_id"] == "w1"

    def test_started_to_done(self, db_conn):
        job_id = self._create_job(db_conn)
        db_conn.execute(
            "UPDATE parse_jobs SET status = 'started' WHERE id = %s", [job_id]
        )
        db_conn.execute(
            """UPDATE parse_jobs
               SET status = 'done', page_count = 5, block_count = 10
               WHERE id = %s""",
            [job_id],
        )
        row = db_conn.execute(
            "SELECT status, page_count, block_count FROM parse_jobs WHERE id = %s",
            [job_id],
        ).fetchone()
        assert row["status"] == "done"
        assert row["page_count"] == 5

    def test_started_to_failed(self, db_conn):
        job_id = self._create_job(db_conn)
        db_conn.execute(
            "UPDATE parse_jobs SET status = 'started' WHERE id = %s", [job_id]
        )
        db_conn.execute(
            """UPDATE parse_jobs
               SET status = 'failed', error_message = 'test error'
               WHERE id = %s""",
            [job_id],
        )
        row = db_conn.execute(
            "SELECT status, error_message FROM parse_jobs WHERE id = %s", [job_id]
        ).fetchone()
        assert row["status"] == "failed"
        assert row["error_message"] == "test error"


class TestWorkerHeartbeats:
    def test_upsert_creates_new(self, db_conn):
        wid = f"test-worker-{uuid.uuid4().hex[:8]}"
        db_conn.execute(
            """INSERT INTO worker_heartbeats (worker_id, hostname, cpu_count)
               VALUES (%s, 'test-host', 8)
               ON CONFLICT (worker_id) DO UPDATE SET hostname = EXCLUDED.hostname""",
            [wid],
        )
        row = db_conn.execute(
            "SELECT * FROM worker_heartbeats WHERE worker_id = %s", [wid]
        ).fetchone()
        assert row is not None
        assert row["cpu_count"] == 8

    def test_upsert_updates_existing(self, db_conn):
        wid = f"test-worker-{uuid.uuid4().hex[:8]}"
        db_conn.execute(
            """INSERT INTO worker_heartbeats (worker_id, hostname, cpu_count, pdfs_completed)
               VALUES (%s, 'host', 8, 0)""",
            [wid],
        )
        db_conn.execute(
            """INSERT INTO worker_heartbeats (worker_id, hostname, cpu_count, pdfs_completed)
               VALUES (%s, 'host', 8, 5)
               ON CONFLICT (worker_id) DO UPDATE SET pdfs_completed = EXCLUDED.pdfs_completed""",
            [wid],
        )
        row = db_conn.execute(
            "SELECT pdfs_completed FROM worker_heartbeats WHERE worker_id = %s", [wid]
        ).fetchone()
        assert row["pdfs_completed"] == 5


# ── RAG tables (processed_chunks + chunk_embeddings) ──────────────────────

def _create_chunk_job(db_conn) -> tuple[str, str]:
    """Helper: create a parse_job + chunk_job, return (chunk_job_id, pdf_id)."""
    pdf_id = uuid.uuid4()
    row = db_conn.execute(
        """INSERT INTO parse_jobs (pdf_id, gcs_path, gcs_bucket, status)
           VALUES (%s, 'test/rag.pdf', 'bucket', 'done')
           RETURNING id""",
        [str(pdf_id)],
    ).fetchone()
    parse_job_id = str(row["id"])

    row = db_conn.execute(
        """INSERT INTO chunk_jobs (parse_job_id, pdf_id, input_gcs_path, status)
           VALUES (%s, %s, 'test/rag.json', 'done')
           RETURNING id""",
        [parse_job_id, str(pdf_id)],
    ).fetchone()
    return str(row["id"]), str(pdf_id)


def _insert_processed_chunk(db_conn, chunk_job_id: str, pdf_id: str, chunk_index: int = 0, search_text: str = "test search text for embedding") -> str:
    """Helper: insert a processed_chunks row and return chunk_id."""
    row = db_conn.execute(
        """INSERT INTO processed_chunks
               (pdf_id, chunk_job_id, chunk_index, chunk_type,
                display_text, search_text, page_start, page_end)
           VALUES (%s, %s, %s, 'narrative', 'display text', %s, 1, 2)
           RETURNING chunk_id""",
        [pdf_id, chunk_job_id, chunk_index, search_text],
    ).fetchone()
    return str(row["chunk_id"])


class TestRagTables:
    def test_tables_created(self, db_conn):
        """Both processed_chunks and chunk_embeddings tables exist."""
        for table_name in ("processed_chunks", "chunk_embeddings"):
            row = db_conn.execute(
                """SELECT EXISTS (
                       SELECT FROM information_schema.tables
                       WHERE table_schema = 'public' AND table_name = %s
                   ) AS exists""",
                [table_name],
            ).fetchone()
            assert row["exists"] is True, f"Table {table_name} does not exist"

    def test_tsvector_auto_populates(self, db_conn):
        """Inserting a chunk auto-populates the tsv generated column."""
        cj_id, pdf_id = _create_chunk_job(db_conn)
        chunk_id = _insert_processed_chunk(
            db_conn, cj_id, pdf_id,
            search_text="പദ്ധതി budget allocation for construction",
        )
        row = db_conn.execute(
            "SELECT tsv FROM processed_chunks WHERE chunk_id = %s",
            [chunk_id],
        ).fetchone()
        assert row["tsv"] is not None
        tsv_str = str(row["tsv"])
        assert "budget" in tsv_str

    def test_unique_constraint_rejects_duplicate(self, db_conn):
        """Same (chunk_job_id, chunk_index) pair is rejected."""
        cj_id, pdf_id = _create_chunk_job(db_conn)
        _insert_processed_chunk(db_conn, cj_id, pdf_id, chunk_index=0)
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_processed_chunk(db_conn, cj_id, pdf_id, chunk_index=0)

    def test_cascade_delete(self, db_conn):
        """Deleting a processed_chunk cascades to chunk_embeddings."""
        cj_id, pdf_id = _create_chunk_job(db_conn)
        chunk_id = _insert_processed_chunk(db_conn, cj_id, pdf_id)

        fake_vector = "[" + ",".join(["0.1"] * 1024) + "]"
        db_conn.execute(
            """INSERT INTO chunk_embeddings (chunk_id, embed_model, embedding)
               VALUES (%s, 'test-model', %s::vector)""",
            [chunk_id, fake_vector],
        )

        row = db_conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunk_embeddings WHERE chunk_id = %s",
            [chunk_id],
        ).fetchone()
        assert row["cnt"] == 1

        db_conn.execute(
            "DELETE FROM processed_chunks WHERE chunk_id = %s", [chunk_id]
        )

        row = db_conn.execute(
            "SELECT COUNT(*) AS cnt FROM chunk_embeddings WHERE chunk_id = %s",
            [chunk_id],
        ).fetchone()
        assert row["cnt"] == 0

    def test_trgm_similarity_query(self, db_conn):
        """pg_trgm index supports similarity queries on search_text."""
        cj_id, pdf_id = _create_chunk_job(db_conn)
        _insert_processed_chunk(
            db_conn, cj_id, pdf_id,
            search_text="stadium electrification project in thrissur district",
        )
        rows = db_conn.execute(
            """SELECT chunk_id, similarity(search_text, %s) AS sim
               FROM processed_chunks
               WHERE search_text %% %s
               ORDER BY sim DESC LIMIT 5""",
            ["stadium electrification thrissur", "stadium electrification thrissur"],
        ).fetchall()
        assert len(rows) >= 1


class TestIdempotentEmbedding:
    def test_rerun_skips_existing(self, db_conn):
        """Second insert with same (chunk_job_id, chunk_index) raises UniqueViolation,
        confirming the constraint works for idempotent re-runs."""
        cj_id, pdf_id = _create_chunk_job(db_conn)
        _insert_processed_chunk(db_conn, cj_id, pdf_id, chunk_index=5)

        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_processed_chunk(db_conn, cj_id, pdf_id, chunk_index=5)
