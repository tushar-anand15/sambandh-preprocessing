#!/usr/bin/env python3
"""Background service that records metrics snapshots and cleans up stale data.

Run as: python -m sambandh.metrics_recorder

Every SNAPSHOT_INTERVAL seconds (default 300 = 5 minutes):
1. Snapshot worker_heartbeats → worker_metrics_history
2. Record throughput per machine → throughput_snapshots
3. Reset stuck jobs (started > 20 min ago) back to pending
4. Delete stale worker heartbeats (no update > 10 min)
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from sambandh import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SNAPSHOT_INTERVAL = int(os.environ.get("SNAPSHOT_INTERVAL", 300))
STUCK_JOB_TIMEOUT_MINUTES = int(os.environ.get("STUCK_JOB_TIMEOUT_MINUTES", 20))
STALE_HEARTBEAT_TIMEOUT_MINUTES = int(os.environ.get("STALE_HEARTBEAT_TIMEOUT_MINUTES", 10))

_previous_completed: dict[str, int] = {}
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum}, shutting down...")
    _shutdown = True


def snapshot_worker_metrics():
    """Copy current heartbeat data to history table."""
    heartbeats = db.get_active_heartbeats()
    now = datetime.now(timezone.utc)
    
    for hb in heartbeats:
        worker_id = hb["worker_id"]
        machine_name = hb.get("machine_name") or _extract_machine_name(worker_id)
        
        prev_completed = _previous_completed.get(worker_id, 0)
        current_completed = hb["pdfs_completed"] or 0
        pdfs_in_interval = max(0, current_completed - prev_completed)
        _previous_completed[worker_id] = current_completed
        
        db.insert_metrics_snapshot(
            worker_id=worker_id,
            hostname=hb.get("hostname"),
            machine_name=machine_name,
            cpu_percent=hb.get("cpu_percent") or 0,
            mem_percent=hb.get("mem_percent") or 0,
            load_avg=hb.get("load_avg") or 0,
            pdfs_completed_total=current_completed,
            pdfs_in_interval=pdfs_in_interval,
        )
    
    logger.info(f"Recorded metrics for {len(heartbeats)} active workers")


def snapshot_throughput():
    """Record throughput per machine for the last interval."""
    now = datetime.now(timezone.utc)
    interval_end = now
    interval_start = now - timedelta(seconds=SNAPSHOT_INTERVAL)
    
    interval_minutes = SNAPSHOT_INTERVAL // 60
    throughput_data = db.get_throughput_by_machine(since_minutes=interval_minutes)
    
    total_completed = 0
    for row in throughput_data:
        machine_name = row.get("machine_name") or "unknown"
        pdfs_completed = row["pdfs_completed"]
        avg_parse_time = row.get("avg_parse_time")
        total_completed += pdfs_completed
        
        db.insert_throughput_snapshot(
            interval_start=interval_start,
            interval_end=interval_end,
            machine_name=machine_name,
            pdfs_completed=pdfs_completed,
            avg_parse_time=avg_parse_time,
        )
    
    if total_completed > 0 or throughput_data:
        db.insert_throughput_snapshot(
            interval_start=interval_start,
            interval_end=interval_end,
            machine_name=None,
            pdfs_completed=total_completed,
            avg_parse_time=None,
        )
    
    logger.info(f"Recorded throughput: {total_completed} PDFs in last {interval_minutes} min")


def cleanup_stale_data():
    """Reset stuck jobs and remove dead worker heartbeats."""
    stuck_jobs = db.reset_stuck_jobs(timeout_minutes=STUCK_JOB_TIMEOUT_MINUTES)
    for job in stuck_jobs:
        logger.warning(
            f"Reset stuck job {job['id']} "
            f"(path: {job['gcs_path']}, was on: {job['worker_id']})"
        )
    
    if stuck_jobs:
        logger.info(f"Reset {len(stuck_jobs)} stuck jobs back to pending")
    
    dead_workers = db.delete_stale_heartbeats(timeout_minutes=STALE_HEARTBEAT_TIMEOUT_MINUTES)
    for w in dead_workers:
        worker_id = w.get("worker_id", "unknown")
        if worker_id in _previous_completed:
            del _previous_completed[worker_id]
        logger.info(f"Removed dead worker heartbeat: {worker_id}")
    
    if dead_workers:
        logger.info(f"Cleaned up {len(dead_workers)} stale heartbeats")


def _extract_machine_name(worker_id: str) -> str:
    """Extract machine name from worker_id like 'worker-hostname-pid'."""
    parts = worker_id.split("-")
    if len(parts) >= 3:
        return "-".join(parts[1:-1])
    return worker_id


def run_snapshot_cycle():
    """Run one complete snapshot and cleanup cycle."""
    try:
        snapshot_worker_metrics()
    except Exception as e:
        logger.error(f"Failed to snapshot worker metrics: {e}")
    
    try:
        snapshot_throughput()
    except Exception as e:
        logger.error(f"Failed to snapshot throughput: {e}")
    
    try:
        cleanup_stale_data()
    except Exception as e:
        logger.error(f"Failed to cleanup stale data: {e}")


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    
    logger.info(f"Metrics recorder starting (interval={SNAPSHOT_INTERVAL}s)")
    logger.info(f"Stuck job timeout: {STUCK_JOB_TIMEOUT_MINUTES} min")
    logger.info(f"Stale heartbeat timeout: {STALE_HEARTBEAT_TIMEOUT_MINUTES} min")
    
    run_snapshot_cycle()
    
    while not _shutdown:
        time.sleep(SNAPSHOT_INTERVAL)
        if _shutdown:
            break
        run_snapshot_cycle()
    
    logger.info("Metrics recorder stopped")


if __name__ == "__main__":
    main()
