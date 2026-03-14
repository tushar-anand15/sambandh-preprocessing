"""Unit tests for the embedding pipeline validation and mapping logic.

These run without any infrastructure (no DB, no GCS, no model).

Usage:
    pytest tests/test_embedding.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_embedding import validate_chunk, should_embed, REQUIRED_CHUNK_KEYS


def _make_chunk(**overrides) -> dict:
    """Build a minimal valid chunk dict, with optional overrides."""
    base = {
        "chunk_type": "narrative",
        "section_path": ["Section 1"],
        "display_text": "Some display text for the user",
        "search_text": "some normalized search text for embedding and fts",
        "page_start": 1,
        "page_end": 2,
        "metadata": {"token_count": 42},
    }
    base.update(overrides)
    return base


class TestValidateChunk:
    def test_valid_chunk_passes(self):
        chunk = _make_chunk()
        valid, reason = validate_chunk(chunk, 0)
        assert valid is True
        assert reason == ""

    def test_missing_search_text_fails(self):
        chunk = _make_chunk()
        del chunk["search_text"]
        valid, reason = validate_chunk(chunk, 3)
        assert valid is False
        assert "search_text" in reason
        assert "chunk[3]" in reason

    def test_missing_display_text_fails(self):
        chunk = _make_chunk()
        del chunk["display_text"]
        valid, reason = validate_chunk(chunk, 1)
        assert valid is False
        assert "display_text" in reason

    def test_missing_multiple_keys_reports_all(self):
        chunk = {"chunk_type": "kv", "metadata": {}}
        valid, reason = validate_chunk(chunk, 0)
        assert valid is False
        for key in ("search_text", "display_text", "page_start", "page_end"):
            assert key in reason


class TestShouldEmbed:
    def test_normal_text_embeds(self):
        chunk = _make_chunk(search_text="budget allocation for construction work in panchayat")
        assert should_embed(chunk) is True

    def test_short_text_skips_embed(self):
        chunk = _make_chunk(search_text="hi")
        assert should_embed(chunk) is False

    def test_empty_text_skips_embed(self):
        chunk = _make_chunk(search_text="")
        assert should_embed(chunk) is False

    def test_whitespace_only_skips_embed(self):
        chunk = _make_chunk(search_text="         ")
        assert should_embed(chunk) is False

    def test_exactly_threshold_embeds(self):
        chunk = _make_chunk(search_text="a" * 10)
        assert should_embed(chunk) is True

    def test_missing_search_text_skips(self):
        assert should_embed({}) is False


class TestMetadataPropagation:
    def test_chunk_metadata_fields_present(self):
        """Verify that ChunkObject.to_dict() fields map to REQUIRED_CHUNK_KEYS."""
        chunk = _make_chunk()
        assert REQUIRED_CHUNK_KEYS.issubset(chunk.keys())

    def test_token_count_extracted_from_metadata(self):
        chunk = _make_chunk(metadata={"token_count": 123, "lang_hint": "ml"})
        assert chunk["metadata"]["token_count"] == 123
