"""Celery task: parse a single PDF with Docling + Tesseract OCR."""

import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import psutil

from sambandh.celery_app import app
from sambandh import db, gcs
from sambandh.converter import get_converter

MACHINE_NAME = os.environ.get("MACHINE_NAME", socket.gethostname())
HOSTNAME = socket.gethostname()
# Use hostname (container ID in Docker) for unique worker identification
WORKER_ID = f"worker-{MACHINE_NAME}-{HOSTNAME}"

_heartbeat_lock = threading.Lock()
_stats = {"completed": 0, "failed": 0, "current_pdf": None}


def _heartbeat_loop(stop_event: threading.Event, interval: float = 30.0):
    """Background thread that updates worker_heartbeats every `interval` seconds."""
    cpu_count = os.cpu_count() or 1
    proc = psutil.Process()
    proc.cpu_percent()  # First call returns 0, need to prime it
    
    while not stop_event.wait(timeout=interval):
        try:
            with _heartbeat_lock:
                current = _stats.copy()
            
            # Per-process metrics (this worker's actual usage)
            proc_cpu = proc.cpu_percent()
            proc_mem = proc.memory_percent()
            
            db.upsert_heartbeat(
                worker_id=WORKER_ID,
                hostname=HOSTNAME,
                cpu_count=cpu_count,
                cpu_percent=proc_cpu,
                mem_percent=proc_mem,
                load_avg=os.getloadavg()[0],
                pdfs_completed=current["completed"],
                pdfs_failed=current["failed"],
                current_pdf=current["current_pdf"],
                machine_name=MACHINE_NAME,
            )
        except Exception:
            pass


_heartbeat_stop = None
_heartbeat_thread = None


def _ensure_heartbeat():
    global _heartbeat_stop, _heartbeat_thread
    if _heartbeat_thread is None or not _heartbeat_thread.is_alive():
        _heartbeat_stop = threading.Event()
        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop, args=(_heartbeat_stop,), daemon=True
        )
        _heartbeat_thread.start()


class SanityCheckFailed(Exception):
    pass


@app.task(bind=True, queue="ocr", max_retries=2, default_retry_delay=30)
def parse_pdf(self, job_id: str):
    _ensure_heartbeat()

    job = db.get_job(job_id)
    if job is None:
        return
    if job["status"] == "done":
        return

    pdf_name = Path(job["gcs_path"]).name

    with _heartbeat_lock:
        _stats["current_pdf"] = pdf_name

    db.update_job_started(job_id, WORKER_ID)
    start_time = time.time()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        gcs.download_pdf(job["gcs_bucket"], job["gcs_path"], tmp_path)

        converter = get_converter()
        result = converter.convert(str(tmp_path))
        doc = result.document

        markdown = doc.export_to_markdown()

        blocks = []
        for item, level in doc.iterate_items():
            block_info = {"type": type(item).__name__, "level": level}
            if hasattr(item, "text"):
                block_info["text"] = item.text
            if hasattr(item, "prov") and item.prov:
                block_info["page"] = item.prov[0].page_no if item.prov else None
            blocks.append(block_info)

        tables = []
        for table in doc.tables:
            try:
                table_data = table.export_to_dataframe(doc)
                tables.append({
                    "headers": list(table_data.columns),
                    "rows": table_data.values.tolist()[:50],
                    "total_rows": len(table_data),
                })
            except Exception:
                pass

        page_count = doc.num_pages()
        block_count = len(blocks)
        table_count = len(tables)

        if page_count == 0:
            raise SanityCheckFailed(f"page_count=0 for {pdf_name}")
        if block_count == 0:
            raise SanityCheckFailed(f"block_count=0 for {pdf_name}")
        if len(markdown) < 100:
            raise SanityCheckFailed(
                f"markdown too short ({len(markdown)} chars) for {pdf_name}"
            )

        output_data = {
            "pdf_id": str(job["pdf_id"]),
            "filename": pdf_name,
            "gcs_source": job["gcs_path"],
            "metadata": {
                "district": job["district_name"],
                "lb_name": job["lb_name"],
                "lb_type": job["lb_type"],
                "project_no": job["project_no"],
                "project_name": job["project_name"],
                "year": job["year_label"],
            },
            "page_count": page_count,
            "block_count": block_count,
            "table_count": table_count,
            "markdown": markdown,
            "blocks": blocks,
            "tables": tables,
        }

        output_path = gcs.build_output_path(
            year_label=job["year_label"],
            lb_type=job["lb_type"],
            district=job["district_name"],
            lb_name=job["lb_name"],
            project_no=job["project_no"],
        )
        output_size = gcs.upload_json(output_data, output_path)

        elapsed = time.time() - start_time

        db.update_job_done(
            job_id=job_id,
            parse_time_seconds=elapsed,
            output_gcs_path=output_path,
            page_count=page_count,
            block_count=block_count,
            table_count=table_count,
            output_size_bytes=output_size,
        )

        with _heartbeat_lock:
            _stats["completed"] += 1
            _stats["current_pdf"] = None

    except SanityCheckFailed as exc:
        elapsed = time.time() - start_time
        db.update_job_failed(job_id, str(exc), retry=False)
        with _heartbeat_lock:
            _stats["failed"] += 1
            _stats["current_pdf"] = None

    except Exception as exc:
        elapsed = time.time() - start_time
        is_transient = _is_transient(exc)
        db.update_job_failed(job_id, str(exc), retry=is_transient)
        with _heartbeat_lock:
            if is_transient:
                _stats["current_pdf"] = None
            else:
                _stats["failed"] += 1
                _stats["current_pdf"] = None
        if is_transient and self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _is_transient(exc: Exception) -> bool:
    transient_types = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    exc_str = str(type(exc).__name__).lower()
    return isinstance(exc, transient_types) or "timeout" in exc_str or "connection" in exc_str
