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
