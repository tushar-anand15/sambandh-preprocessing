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

ACTIVE_WORKER_THRESHOLD_MINUTES = 5

# IST timezone offset
IST = timezone(timedelta(hours=5, minutes=30))


def to_ist(dt):
    """Convert datetime to IST and format nicely."""
    if dt is None:
        return "-"
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist_dt = dt.astimezone(IST)
    return ist_dt.strftime("%d %b %H:%M:%S")


def to_ist_short(dt):
    """Convert datetime to IST, short format for charts."""
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist_dt = dt.astimezone(IST)
    return ist_dt.strftime("%H:%M")


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

# Show current time in IST
now_ist = datetime.now(IST)
st.sidebar.caption(f"Time: {now_ist.strftime('%d %b %Y %H:%M:%S')} IST")

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
        st.error(f"Warning: {bad_count} completed jobs have page_count=0 or block_count=0")

    st.divider()

    # Throughput metrics
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
            eta_time = now_ist + timedelta(hours=eta_h)
            st.progress(done / max(total, 1), text=f"{done}/{total} done — ETA: {eta_h:.1f}h ({eta_time.strftime('%d %b %H:%M')} IST)")
        elif total > 0:
            st.progress(done / max(total, 1), text=f"{done}/{total} done")

    st.divider()

    # Throughput over time chart (from snapshots if available, else from parse_jobs)
    st.subheader("Throughput Over Time")
    
    throughput_df = query_df(
        """SELECT 
               recorded_at,
               COALESCE(machine_name, 'total') AS machine,
               pdfs_completed
           FROM throughput_snapshots 
           WHERE machine_name IS NOT NULL
             AND recorded_at > now() - interval '2 hours'
           ORDER BY recorded_at"""
    )
    
    if not throughput_df.empty:
        throughput_df["time_ist"] = throughput_df["recorded_at"].apply(to_ist_short)
        # Aggregate duplicates before pivoting
        throughput_agg = throughput_df.groupby(["time_ist", "machine"])["pdfs_completed"].sum().reset_index()
        throughput_pivot = throughput_agg.pivot(
            index="time_ist", columns="machine", values="pdfs_completed"
        ).fillna(0)
        st.line_chart(throughput_pivot)
    else:
        # Fallback: compute from parse_jobs directly
        timeline_df = query_df(
            """SELECT 
                   date_trunc('minute', completed_at) AS minute, 
                   count(*) AS completed
               FROM parse_jobs 
               WHERE status = 'done' AND completed_at IS NOT NULL
                 AND completed_at > now() - interval '2 hours'
               GROUP BY 1 ORDER BY 1"""
        )
        if not timeline_df.empty:
            timeline_df["time_ist"] = timeline_df["minute"].apply(to_ist_short)
            st.line_chart(timeline_df.set_index("time_ist")["completed"])
        else:
            st.info("No throughput data yet")

    # Cumulative completion chart
    st.subheader("Cumulative Completion")
    cumulative_df = query_df(
        """SELECT date_trunc('minute', completed_at) AS minute, count(*) AS completed
           FROM parse_jobs WHERE status = 'done' AND completed_at IS NOT NULL
           GROUP BY 1 ORDER BY 1"""
    )
    if not cumulative_df.empty:
        cumulative_df["cumulative"] = cumulative_df["completed"].cumsum()
        cumulative_df["time_ist"] = cumulative_df["minute"].apply(to_ist_short)
        st.line_chart(cumulative_df.set_index("time_ist")["cumulative"])

# ---------------------------------------------------------------------------
# PAGE 2: Machines
# ---------------------------------------------------------------------------
elif page == "Machines":
    st.title("Worker Machines")

    # Only show active workers (heartbeat within threshold)
    workers_df = query_df(
        """SELECT worker_id, hostname, machine_name, cpu_count, cpu_percent, mem_percent,
                  load_avg, pdfs_completed, pdfs_failed, current_pdf,
                  started_at, last_seen
           FROM worker_heartbeats 
           WHERE last_seen > now() - interval '%s minutes'
           ORDER BY machine_name, worker_id""",
        [ACTIVE_WORKER_THRESHOLD_MINUTES],
    )

    # Check for recently dead workers (for warning)
    dead_workers = query(
        """SELECT worker_id, last_seen 
           FROM worker_heartbeats 
           WHERE last_seen <= now() - interval '%s minutes'
             AND last_seen > now() - interval '30 minutes'""",
        [ACTIVE_WORKER_THRESHOLD_MINUTES],
    )
    if dead_workers:
        dead_ids = [w["worker_id"].split("-")[-1][:8] for w in dead_workers]
        st.warning(f"Recently inactive workers: {', '.join(dead_ids)}")

    if workers_df.empty:
        st.info("No active workers reporting.")
    else:
        # Format last_seen as IST
        workers_df["last_seen_ist"] = workers_df["last_seen"].apply(to_ist)
        
        # Summary by machine
        machine_summary = workers_df.groupby("machine_name").agg({
            "worker_id": "count",
            "cpu_percent": "mean",
            "mem_percent": "mean",
            "pdfs_completed": "sum",
            "pdfs_failed": "sum",
        }).rename(columns={
            "worker_id": "workers",
            "cpu_percent": "avg_cpu",
            "mem_percent": "avg_mem",
        })
        
        st.subheader("Summary by Machine")
        st.dataframe(
            machine_summary,
            column_config={
                "avg_cpu": st.column_config.NumberColumn("Avg CPU%", format="%.1f"),
                "avg_mem": st.column_config.NumberColumn("Avg Mem%", format="%.1f"),
            },
            use_container_width=True,
        )

        st.divider()

        # Worker details table
        st.subheader("Active Workers")
        display_df = workers_df[
            ["worker_id", "machine_name", "cpu_percent", "mem_percent", "load_avg",
             "pdfs_completed", "pdfs_failed", "current_pdf", "last_seen_ist"]
        ].copy()
        display_df["worker_id"] = display_df["worker_id"].apply(lambda x: x.split("-")[-1][:12])
        
        st.dataframe(
            display_df,
            column_config={
                "worker_id": "Worker",
                "cpu_percent": st.column_config.NumberColumn("CPU%", format="%.1f"),
                "mem_percent": st.column_config.NumberColumn("Mem%", format="%.1f"),
                "load_avg": st.column_config.NumberColumn("Load", format="%.1f"),
                "last_seen_ist": "Last Seen (IST)",
            },
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        # Historical CPU/Memory charts from metrics_history
        st.subheader("Resource Usage Over Time")
        
        metrics_df = query_df(
            """SELECT 
                   recorded_at,
                   machine_name,
                   AVG(cpu_percent) AS cpu_percent,
                   AVG(mem_percent) AS mem_percent
               FROM worker_metrics_history
               WHERE recorded_at > now() - interval '2 hours'
               GROUP BY recorded_at, machine_name
               ORDER BY recorded_at"""
        )
        
        if not metrics_df.empty:
            metrics_df["time_ist"] = metrics_df["recorded_at"].apply(to_ist_short)
            # Aggregate duplicates before pivoting
            cpu_agg = metrics_df.groupby(["time_ist", "machine_name"])["cpu_percent"].mean().reset_index()
            mem_agg = metrics_df.groupby(["time_ist", "machine_name"])["mem_percent"].mean().reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("CPU Usage (%)")
                cpu_pivot = cpu_agg.pivot(
                    index="time_ist", columns="machine_name", values="cpu_percent"
                ).fillna(0)
                st.line_chart(cpu_pivot)
            
            with col2:
                st.caption("Memory Usage (%)")
                mem_pivot = mem_agg.pivot(
                    index="time_ist", columns="machine_name", values="mem_percent"
                ).fillna(0)
                st.line_chart(mem_pivot)
        else:
            # Fallback: current snapshot bar charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("Current CPU Usage (%)")
                cpu_chart_data = workers_df.groupby("machine_name")["cpu_percent"].mean()
                st.bar_chart(cpu_chart_data)
            
            with col2:
                st.caption("Current Memory Usage (%)")
                mem_chart_data = workers_df.groupby("machine_name")["mem_percent"].mean()
                st.bar_chart(mem_chart_data)

        st.divider()

        # Throughput by machine
        st.subheader("PDFs Completed by Machine")
        worker_throughput = query_df(
            """SELECT 
                   COALESCE(wh.machine_name, SPLIT_PART(pj.worker_id, '-', 2)) AS machine_name,
                   count(*) AS done
               FROM parse_jobs pj
               LEFT JOIN worker_heartbeats wh ON wh.worker_id = pj.worker_id
               WHERE pj.status = 'done'
               GROUP BY 1 ORDER BY done DESC"""
        )
        if not worker_throughput.empty:
            st.bar_chart(worker_throughput.set_index("machine_name")["done"])

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

    # Cascading filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        districts = query("SELECT DISTINCT district_name FROM parse_jobs WHERE status='done' ORDER BY 1")
        district_names = [r["district_name"] for r in districts if r["district_name"]]
        selected_district = st.selectbox("District", ["All"] + district_names)
    
    with col2:
        lb_query = "SELECT DISTINCT lb_name FROM parse_jobs WHERE status='done'"
        lb_params = []
        if selected_district != "All":
            lb_query += " AND district_name = %s"
            lb_params.append(selected_district)
        lb_query += " ORDER BY 1"
        lbs = query(lb_query, lb_params)
        lb_names = [r["lb_name"] for r in lbs if r["lb_name"]]
        selected_lb = st.selectbox("LB Name", ["All"] + lb_names)
    
    with col3:
        project_query = "SELECT DISTINCT project_no FROM parse_jobs WHERE status='done'"
        project_params = []
        if selected_district != "All":
            project_query += " AND district_name = %s"
            project_params.append(selected_district)
        if selected_lb != "All":
            project_query += " AND lb_name = %s"
            project_params.append(selected_lb)
        project_query += " ORDER BY 1"
        projects = query(project_query, project_params)
        project_nos = [r["project_no"] for r in projects if r["project_no"]]
        selected_project = st.selectbox("Project No", ["All"] + project_nos)
    
    with col4:
        search_term = st.text_input("Search project name")

    # Build filter conditions
    where_clauses = ["status = 'done'"]
    params = []
    
    if selected_district != "All":
        where_clauses.append("district_name = %s")
        params.append(selected_district)
    if selected_lb != "All":
        where_clauses.append("lb_name = %s")
        params.append(selected_lb)
    if selected_project != "All":
        where_clauses.append("project_no = %s")
        params.append(selected_project)
    if search_term:
        where_clauses.append("project_name ILIKE %s")
        params.append(f"%{search_term}%")
    
    where = " AND ".join(where_clauses)

    # Get total count for pagination
    count_result = query(f"SELECT count(*) AS cnt FROM parse_jobs WHERE {where}", params)
    total_docs = count_result[0]["cnt"] if count_result else 0

    # Pagination controls
    page_size = 20
    total_pages = max(1, (total_docs + page_size - 1) // page_size)
    
    pcol1, pcol2, pcol3, pcol4 = st.columns([1, 1, 1, 1])
    with pcol1:
        if st.button("< Prev", disabled=(st.session_state.get("doc_page", 1) <= 1)):
            st.session_state["doc_page"] = max(1, st.session_state.get("doc_page", 1) - 1)
    with pcol2:
        page_num = st.number_input(
            "Page", 
            min_value=1, 
            max_value=total_pages, 
            value=st.session_state.get("doc_page", 1),
            key="doc_page_input",
            label_visibility="collapsed"
        )
        st.session_state["doc_page"] = page_num
    with pcol3:
        if st.button("Next >", disabled=(st.session_state.get("doc_page", 1) >= total_pages)):
            st.session_state["doc_page"] = min(total_pages, st.session_state.get("doc_page", 1) + 1)
    with pcol4:
        st.caption(f"Page {page_num} of {total_pages}")
    
    offset = (page_num - 1) * page_size
    st.caption(f"Showing {offset + 1}-{min(offset + page_size, total_docs)} of {total_docs:,} documents")

    # Query documents
    docs = query_df(
        f"""SELECT id, pdf_id, district_name, lb_name, lb_type, project_no,
                   project_name, page_count, block_count, table_count,
                   output_gcs_path, parse_time_seconds, gcs_bucket, gcs_path,
                   completed_at
            FROM parse_jobs 
            WHERE {where}
            ORDER BY district_name, lb_name, project_no
            LIMIT %s OFFSET %s""",
        params + [page_size, offset],
    )

    if docs.empty:
        st.info("No completed documents match the filters.")
    else:
        # Format for display
        docs["completed_ist"] = docs["completed_at"].apply(to_ist)
        
        # Clickable table for document selection
        st.subheader("Select a Document")
        
        # Create display dataframe
        display_docs = docs[["district_name", "lb_name", "project_no", "project_name", 
                            "page_count", "parse_time_seconds", "completed_ist"]].copy()
        display_docs.columns = ["District", "LB Name", "Project #", "Project Name", 
                               "Pages", "Parse Time (s)", "Completed (IST)"]
        
        # Use data_editor for selection
        selected_rows = st.dataframe(
            display_docs,
            use_container_width=True,
            hide_index=False,
            column_config={
                "Parse Time (s)": st.column_config.NumberColumn(format="%.1f"),
            },
            on_select="rerun",
            selection_mode="single-row",
        )
        
        # Get selected row
        selected_indices = selected_rows.selection.rows if selected_rows.selection else []
        
        if selected_indices:
            selected_idx = selected_indices[0]
            row = docs.iloc[selected_idx]
            
            st.divider()
            
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
                            pdf_viewer(tmp.name, height=600)
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
                            # Scrollable markdown container
                            markdown_content = data.get("markdown", "_No markdown_")
                            st.markdown(
                                f"""<div style="height: 600px; overflow-y: auto; border: 1px solid #333; 
                                    padding: 15px; border-radius: 5px; background-color: #0e1117;">
                                    {markdown_content}
                                </div>""",
                                unsafe_allow_html=True
                            )
                        with tab_json:
                            st.json(data, expanded=False)
                    except Exception as e:
                        st.error(f"Could not load output JSON: {e}")
                else:
                    st.warning("No output path recorded.")
        else:
            st.info("Click on a row above to view the document details.")

# ---------------------------------------------------------------------------
# Auto-refresh (only for Overview and Machines pages)
# ---------------------------------------------------------------------------
if page != "Document Viewer":
    time.sleep(10)
    st.rerun()
