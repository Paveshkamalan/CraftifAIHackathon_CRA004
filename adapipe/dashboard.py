"""
AdaPipe Minimal Dashboard (Streamlit).
Implements FR-11:
Displays current selected configuration, benchmark comparisons,
live FPS/latency metrics, SLO constraint status, detection counts,
and recent adaptation / guardrail events.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AdaPipe Control Plane",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ AdaPipe Control Plane")
st.caption("Constraint-Aware, Benchmark-Driven, Runtime-Adaptive Control Layer for PipeGen Vision Pipelines")
st.markdown("**Track:** CraftifAI Physical AI Hackathon — Track A: PipeGen | **Author:** Pavesh Kamalan R")

# Paths
results_dir = Path("results")
csv_path = results_dir / "benchmark_results.csv"
sel_path = results_dir / "selection_report.json"
log_path = results_dir / "adaptation_log.jsonl"
md_path = results_dir / "selection_report.md"

# Load selection report
sel_data = {}
if sel_path.exists():
    try:
        with open(sel_path, "r", encoding="utf-8") as f:
            sel_data = json.load(f)
    except Exception:
        pass

slo = sel_data.get("slo", {"min_fps": 15, "max_p95_latency_ms": 250, "min_detection_confidence": 0.45})
selected_candidate = sel_data.get("selected_candidate")
overall_slo = sel_data.get("overall_slo_status", "UNKNOWN")

# Top KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("SLO Min FPS", f"{slo.get('min_fps', 15)} FPS")
with c2:
    st.metric("SLO Max p95 Latency", f"{slo.get('max_p95_latency_ms', 250)} ms")
with c3:
    st.metric("SLO Min Confidence", f"{slo.get('min_detection_confidence', 0.45):.2f}")
with c4:
    color = "normal" if overall_slo == "SLO_MET" else "inverse"
    st.metric("Overall SLO Status", overall_slo)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Candidate Benchmarks",
    "⚙️ Selected Configuration",
    "🔄 Runtime Adaptation Log",
    "📄 Evidence & Markdown Report",
])

with tab1:
    st.subheader("Candidate Benchmarking Results")
    if csv_path.exists() and csv_path.stat().st_size > 0:
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("#### FPS by Candidate")
            st.bar_chart(df.set_index("candidate_id")["fps"])
        with col_b:
            st.write("#### p95 Latency (ms) by Candidate")
            st.bar_chart(df.set_index("candidate_id")["p95_latency_ms"])

        st.write("#### Telemetry Overview")
        st.table(df[["candidate_id", "fps", "p50_latency_ms", "p95_latency_ms", "mean_confidence", "gpu_utilization_percent", "cpu_utilization_percent", "status"]])
    else:
        st.info("No benchmark results found. Run `python -m adapipe.main --mode benchmark`.")

with tab2:
    st.subheader("Configuration Selection Decision")
    if selected_candidate:
        st.success(f"**Selected Candidate ID:** `{selected_candidate.get('id')}`")
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.write(f"- **Inference Interval:** `{selected_candidate.get('inference_interval')}`")
            st.write(f"- **Resolution:** `{selected_candidate.get('resolution')}`")
        with col_y:
            st.write(f"- **ROI Mode:** `{selected_candidate.get('roi_mode')}`")
            st.write(f"- **SLO Compliance:** `{selected_candidate.get('slo_status')}`")
        with col_z:
            st.write(f"- **Normalized Violation Score:** `{selected_candidate.get('violation_score', 0.0):.4f}`")

        st.info(f"**Selection Rationale:**\n{sel_data.get('selection_reason', 'N/A')}")
    else:
        st.warning("No candidate selected yet. Run `python -m adapipe.main --mode select`.")

with tab3:
    st.subheader("Runtime Adaptation Events (Hysteresis & Guardrails)")
    events = []
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass

    if events:
        events_df = pd.DataFrame(events)
        st.dataframe(events_df, use_container_width=True)
        st.caption(f"Total events recorded: {len(events)}")
    else:
        st.info("No runtime adaptation events logged yet. Run `python -m adapipe.main --mode adaptive`.")

with tab4:
    st.subheader("Selection Report Markdown")
    if md_path.exists():
        content = md_path.read_text(encoding="utf-8")
        st.markdown(content)
    else:
        st.info("Markdown report not generated yet. Run `python -m adapipe.main --mode report`.")
