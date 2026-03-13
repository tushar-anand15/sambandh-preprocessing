"""GCS download / upload helpers."""

import json
import os
from pathlib import Path

from google.cloud import storage

GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET", "parsed-sulekha")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def download_pdf(gcs_bucket: str, gcs_path: str, local_path: Path) -> None:
    client = _get_client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(str(local_path))


def upload_json(data: dict, output_gcs_path: str) -> int:
    """Upload parsed JSON to the output bucket. Returns size in bytes."""
    client = _get_client()
    bucket = client.bucket(GCS_OUTPUT_BUCKET)
    blob = bucket.blob(output_gcs_path)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    blob.upload_from_string(payload, content_type="application/json")
    return len(payload.encode("utf-8"))


def build_output_path(year_label: str, lb_type: str, district: str, lb_name: str, project_no: str) -> str:
    safe = lambda s: s.replace(" ", "_").replace("/", "_") if s else "unknown"
    return f"{safe(year_label)}/{safe(lb_type)}/{safe(district)}/{safe(lb_name)}/{safe(project_no)}.json"


def download_json_from_gcs(output_gcs_path: str) -> dict:
    """Download a parsed JSON from the output bucket. Used by the dashboard viewer."""
    client = _get_client()
    bucket = client.bucket(GCS_OUTPUT_BUCKET)
    blob = bucket.blob(output_gcs_path)
    return json.loads(blob.download_as_text())
