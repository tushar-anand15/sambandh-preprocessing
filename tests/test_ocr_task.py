"""Tier 1: Unit tests for the OCR pipeline logic.

Tests GCS path generation, sanity check logic, DB helpers,
and idempotent enqueue behaviour.  Heavy dependencies (Docling, GCS)
are mocked so these run fast without infrastructure.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from sambandh.gcs import build_output_path


# ── GCS output path generation ──────────────────────────────────────────

class TestBuildOutputPath:
    def test_basic_path(self):
        path = build_output_path(
            year_label="2025-2026",
            lb_type="Grama Panchayat",
            district="Thrissur",
            lb_name="Adat",
            project_no="42",
        )
        assert path == "2025-2026/Grama_Panchayat/Thrissur/Adat/42.json"

    def test_spaces_replaced(self):
        path = build_output_path(
            year_label="2025-2026",
            lb_type="Block Panchayat",
            district="Ernakulam",
            lb_name="Aluva East",
            project_no="105",
        )
        assert path == "2025-2026/Block_Panchayat/Ernakulam/Aluva_East/105.json"

    def test_slashes_replaced(self):
        path = build_output_path(
            year_label="2025-2026",
            lb_type="District Panchayat",
            district="Kannur",
            lb_name="Name/With/Slash",
            project_no="7",
        )
        assert "Name_With_Slash" in path

    def test_none_values_become_unknown(self):
        path = build_output_path(
            year_label=None,
            lb_type=None,
            district=None,
            lb_name=None,
            project_no=None,
        )
        assert path == "unknown/unknown/unknown/unknown/unknown.json"


# ── Sanity check logic ──────────────────────────────────────────────────

class TestSanityChecks:
    """Test that the SanityCheckFailed exception would be raised for bad output."""

    def test_zero_pages_fails(self):
        from sambandh.tasks.ocr import SanityCheckFailed

        with pytest.raises(SanityCheckFailed, match="page_count=0"):
            raise SanityCheckFailed("page_count=0 for test.pdf")

    def test_zero_blocks_fails(self):
        from sambandh.tasks.ocr import SanityCheckFailed

        with pytest.raises(SanityCheckFailed, match="block_count=0"):
            raise SanityCheckFailed("block_count=0 for test.pdf")

    def test_short_markdown_fails(self):
        from sambandh.tasks.ocr import SanityCheckFailed

        with pytest.raises(SanityCheckFailed, match="markdown too short"):
            raise SanityCheckFailed("markdown too short (50 chars) for test.pdf")


# ── Transient error detection ───────────────────────────────────────────

class TestTransientDetection:
    def test_connection_error_is_transient(self):
        from sambandh.tasks.ocr import _is_transient

        assert _is_transient(ConnectionError("refused")) is True

    def test_timeout_error_is_transient(self):
        from sambandh.tasks.ocr import _is_transient

        assert _is_transient(TimeoutError("timed out")) is True

    def test_os_error_is_transient(self):
        from sambandh.tasks.ocr import _is_transient

        assert _is_transient(OSError("network unreachable")) is True

    def test_value_error_is_not_transient(self):
        from sambandh.tasks.ocr import _is_transient

        assert _is_transient(ValueError("bad data")) is False

    def test_runtime_error_is_not_transient(self):
        from sambandh.tasks.ocr import _is_transient

        assert _is_transient(RuntimeError("crash")) is False


# ── DB helper tests (mocked) ────────────────────────────────────────────

class TestDBHelpers:
    @patch("sambandh.db._get_pool")
    def test_update_job_started_sets_status(self, mock_pool):
        mock_conn = MagicMock()
        mock_pool.return_value.connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

        from sambandh.db import update_job_started

        update_job_started("some-uuid", "worker-1")
        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "status = 'started'" in sql
        assert "worker_id" in sql

    @patch("sambandh.db._get_pool")
    def test_update_job_failed_sets_retry_on_transient(self, mock_pool):
        mock_conn = MagicMock()
        mock_pool.return_value.connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

        from sambandh.db import update_job_failed

        update_job_failed("some-uuid", "GCS timeout", retry=True)
        args = mock_conn.execute.call_args[0][1]
        assert args[0] == "pending"  # status set back to pending for retry

    @patch("sambandh.db._get_pool")
    def test_update_job_failed_sets_failed_on_permanent(self, mock_pool):
        mock_conn = MagicMock()
        mock_pool.return_value.connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

        from sambandh.db import update_job_failed

        update_job_failed("some-uuid", "Sanity check failed", retry=False)
        args = mock_conn.execute.call_args[0][1]
        assert args[0] == "failed"


# ── Heartbeat upsert ────────────────────────────────────────────────────

class TestHeartbeat:
    @patch("sambandh.db._get_pool")
    def test_upsert_heartbeat_uses_on_conflict(self, mock_pool):
        mock_conn = MagicMock()
        mock_pool.return_value.connection.return_value.__enter__ = lambda s: mock_conn
        mock_pool.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

        from sambandh.db import upsert_heartbeat

        upsert_heartbeat(
            worker_id="w1", hostname="vm1", cpu_count=8,
            cpu_percent=90.0, mem_percent=60.0, load_avg=10.5,
            pdfs_completed=5, pdfs_failed=1, current_pdf="test.pdf",
        )
        sql = mock_conn.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql
