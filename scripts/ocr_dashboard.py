#!/usr/bin/env python3
"""Streamlit dashboard for monitoring distributed OCR processing."""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg
from psycopg_pool import ConnectionPool
import streamlit as st
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://sambandh:sambandh@localhost:5433/sambandh"
)


@st.cache_resource
def get_conn_pool():
    return ConnectionPool(
        DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"),
        min_size=1,
        max_size=2,
        kwargs={"row_factory": dict_row},
    )


def query(sql: str, params=None) -> list[dict]:
    pool = get_conn_pool()
    with pool.connection() as conn:
        return conn.execute(sql, params or []).fetchall()


def query_df(sql: str, params=None) -> pd.DataFrame:
    rows = query(sql, params)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sambandh OCR Dashboard", layout="wide")

page = st.sidebar.radio("Navigate", ["Overview", "Machines", "Document Viewer"])

# ---------------------------------------------------------------------------
# PAGE 1: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("OCR Processing Overview")

    status_counts = {
        r["status"]: r["cnt"]
        for r in query("SELECT status, count(*) AS cnt FROM parse_jobs GROUP BY status")
    }
    total = sum(status_counts.values())
    done = status_counts.get("done", 0)
    pending = status_counts.get("pending", 0)
    started = status_counts.get("started", 0)
    failed = status_counts.get("failed", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", f"{total:,}")
    c2.metric("Pending", f"{pending:,}")
    c3.metric("In Progress", f"{started:,}")
    c4.metric("Done", f"{done:,}", delta=None)
    c5.metric("Failed", f"{failed:,}")

    # Sanity alerts
    sanity_issues = query(
        """SELECT count(*) AS cnt FROM parse_jobs
           WHERE status = 'done' AND (page_count = 0 OR block_count = 0 OR page_count IS NULL)"""
    )
    bad_count = sanity_issues[0]["cnt"] if sanity_issues else 0
    if bad_count > 0:
        st.error(f"⚠ {bad_count} completed jobs have page_count=0 or block_count=0 — check Document Viewer")

    st.divider()

    # Throughput
    now = datetime.now(timezone.utc)
    throughput_rows = query(
        """SELECT
               count(*) FILTER (WHERE completed_at > %s) AS last_5m,
               count(*) FILTER (WHERE completed_at > %s) AS last_30m,
               count(*) FILTER (WHERE completed_at IS NOT NULL) AS all_time,
               min(completed_at) AS first_done,
               max(completed_at) AS last_done
           FROM parse_jobs WHERE status = 'done'""",
        [now - timedelta(minutes=5), now - timedelta(minutes=30)],
    )
    if throughput_rows:
        t = throughput_rows[0]
        last_5m = t["last_5m"] or 0
        last_30m = t["last_30m"] or 0
        all_done = t["all_time"] or 0
        first_done = t["first_done"]

        rate_5m = last_5m * 12  # per hour
        rate_30m = last_30m * 2

        if first_done and all_done > 0:
            elapsed_h = max((now - first_done).total_seconds() / 3600, 0.001)
            rate_overall = all_done / elapsed_h
        else:
            rate_overall = 0

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("PDFs/hr (5min)", f"{rate_5m:.0f}")
        tc2.metric("PDFs/hr (30min)", f"{rate_30m:.0f}")
        tc3.metric("PDFs/hr (overall)", f"{rate_overall:.0f}")

        remaining = total - done - failed
        if rate_overall > 0 and remaining > 0:
            eta_h = remaining / rate_overall
            st.progress(done / max(total, 1), text=f"{done}/{total} done — ETA: {eta_h:.1f} hours")
        elif total > 0:
            st.progress(done / max(total, 1), text=f"{done}/{total} done")

    st.divider()

    # Timeline chart
    timeline_df = query_df(
        """SELECT date_trunc('minute', completed_at) AS minute, count(*) AS completed
           FROM parse_jobs WHERE status = 'done' AND completed_at IS NOT NULL
           GROUP BY 1 ORDER BY 1"""
    )
    if not timeline_df.empty:
        timeline_df["cumulative"] = timeline_df["completed"].cumsum()
        st.subheader("Completion Timeline")
        st.line_chart(timeline_df.set_index("minute")["cumulative"])

# ---------------------------------------------------------------------------
# PAGE 2: Machines
# ---------------------------------------------------------------------------
elif page == "Machines":
    st.title("Worker Machines")

    workers_df = query_df(
        """SELECT worker_id, hostname, cpu_count, cpu_percent, mem_percent,
                  load_avg, pdfs_completed, pdfs_failed, current_pdf,
                  started_at, last_seen
           FROM worker_heartbeats ORDER BY worker_id"""
    )

    if workers_df.empty:
        st.info("No workers have reported in yet.")
    else:
        now = datetime.now(timezone.utc)
        workers_df["last_seen_dt"] = pd.to_datetime(workers_df["last_seen"], utc=True)
        workers_df["age_s"] = (now - workers_df["last_seen_dt"]).dt.total_seconds()

        dead_mask = workers_df["age_s"] > 300
        if dead_mask.any():
            dead = workers_df[dead_mask]["worker_id"].tolist()
            st.error(f"⚠ Dead workers (no heartbeat >5min): {', '.join(dead)}")

        st.dataframe(
            workers_df[
                ["worker_id", "hostname", "cpu_percent", "mem_percent", "load_avg",
                 "pdfs_completed", "pdfs_failed", "current_pdf", "last_seen"]
            ],
            column_config={
                "cpu_percent": st.column_config.NumberColumn("CPU%", format="%.1f"),
                "mem_percent": st.column_config.NumberColumn("Mem%", format="%.1f"),
                "load_avg": st.column_config.NumberColumn("Load", format="%.1f"),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        # CPU and Memory charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CPU Usage (%)")
            cpu_chart_data = workers_df[["worker_id", "cpu_percent"]].set_index("worker_id")
            st.bar_chart(cpu_chart_data, color="#ff6b6b")
        
        with col2:
            st.subheader("Memory Usage (%)")
            mem_chart_data = workers_df[["worker_id", "mem_percent"]].set_index("worker_id")
            st.bar_chart(mem_chart_data, color="#4ecdc4")

        st.divider()

        # Per-machine throughput
        worker_throughput = query_df(
            """SELECT worker_id, count(*) AS done
               FROM parse_jobs WHERE status = 'done'
               GROUP BY worker_id ORDER BY done DESC"""
        )
        if not worker_throughput.empty:
            st.subheader("PDFs Completed per Worker")
            st.bar_chart(worker_throughput.set_index("worker_id")["done"])

    # Queue depth from Redis
    try:
        import redis

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6380/0")
        r = redis.from_url(redis_url)
        queue_len = r.llen("ocr")
        st.metric("Celery Queue Depth (ocr)", queue_len)
    except Exception:
        st.caption("Could not read Redis queue depth")

# ---------------------------------------------------------------------------
# PAGE 3: Document Viewer
# ---------------------------------------------------------------------------
elif page == "Document Viewer":
    st.title("Document Viewer")

    col_filter, _ = st.columns([1, 2])
    with col_filter:
        districts = query("SELECT DISTINCT district_name FROM parse_jobs WHERE status='done' ORDER BY 1")
        district_names = [r["district_name"] for r in districts]
        selected_district = st.selectbox("District", ["All"] + district_names)

    where = "WHERE status = 'done'"
    params = []
    if selected_district != "All":
        where += " AND district_name = %s"
        params.append(selected_district)

    docs = query_df(
        f"""SELECT id, pdf_id, district_name, lb_name, lb_type, project_no,
                   project_name, page_count, block_count, table_count,
                   output_gcs_path, parse_time_seconds, gcs_bucket, gcs_path
            FROM parse_jobs {where}
            ORDER BY district_name, lb_name, project_no
            LIMIT 500""",
        params,
    )

    if docs.empty:
        st.info("No completed documents to display.")
    else:
        display_col = (
            docs["district_name"].fillna("")
            + " / " + docs["lb_name"].fillna("")
            + " / #" + docs["project_no"].fillna("")
        )
        selected_idx = st.selectbox("Select document", range(len(docs)), format_func=lambda i: display_col.iloc[i])
        row = docs.iloc[selected_idx]

        left, right = st.columns(2)

        with left:
            st.subheader("PDF")
            st.caption(f"Source: gs://{row.get('gcs_bucket', '')}/{row.get('gcs_path', 'N/A')}")
            try:
                from streamlit_pdf_viewer import pdf_viewer
                from sambandh.gcs import download_pdf as _dl
                import tempfile

                if row.get("gcs_bucket") and row.get("gcs_path"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        _dl(row["gcs_bucket"], row["gcs_path"], tmp.name)
                        pdf_viewer(tmp.name, height=700)
            except ImportError:
                st.warning("Install `streamlit-pdf-viewer` for inline PDF rendering.")
            except Exception as e:
                st.error(f"Could not load PDF: {e}")

        with right:
            st.subheader("Extracted Output")
            st.caption(
                f"Pages: {row['page_count']} | Blocks: {row['block_count']} | "
                f"Tables: {row['table_count']} | Time: {row['parse_time_seconds']:.1f}s"
            )
            if row.get("output_gcs_path"):
                try:
                    from sambandh.gcs import download_json_from_gcs

                    data = download_json_from_gcs(row["output_gcs_path"])
                    tab_md, tab_json = st.tabs(["Markdown", "Raw JSON"])
                    with tab_md:
                        st.markdown(data.get("markdown", "_No markdown_"))
                    with tab_json:
                        st.json(data)
                except Exception as e:
                    st.error(f"Could not load output JSON: {e}")
            else:
                st.warning("No output path recorded.")

# ---------------------------------------------------------------------------
# Auto-refresh (only for Overview and Machines pages)
# ---------------------------------------------------------------------------
if page != "Document Viewer":
    time.sleep(10)
    st.rerun()
