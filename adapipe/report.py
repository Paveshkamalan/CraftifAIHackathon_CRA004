"""
Report generator for AdaPipe.
Transforms CSV benchmark results, JSON selection decisions, and JSONL adaptation events
into human-readable Markdown evidence reports (results/selection_report.md).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_markdown_report(
    selection_report_path: str | Path = "results/selection_report.json",
    benchmark_csv_path: str | Path = "results/benchmark_results.csv",
    adaptation_log_path: str | Path = "results/adaptation_log.jsonl",
    output_md_path: str | Path = "results/selection_report.md",
) -> Path:
    """
    Generate the formal selection_report.md summarizing all benchmarking,
    selection, and runtime adaptation findings.
    """
    sel_path = Path(selection_report_path)
    csv_path = Path(benchmark_csv_path)
    log_path = Path(adaptation_log_path)
    out_md = Path(output_md_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    selection_data: Dict[str, Any] = {}
    if sel_path.exists():
        with open(sel_path, "r", encoding="utf-8") as f:
            selection_data = json.load(f)

    bench_df = pd.DataFrame()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            bench_df = pd.read_csv(csv_path)
        except Exception:
            pass

    adaptation_events: List[Dict[str, Any]] = []
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        adaptation_events.append(json.loads(line))
                    except Exception:
                        pass

    project_name = selection_data.get("project", "AdaPipe")
    use_case = selection_data.get("use_case", "person_detection_entry")
    input_video = selection_data.get("input_video", "demo/sample_video.mp4")
    slo = selection_data.get("slo", {})
    overall_status = selection_data.get("overall_slo_status", "UNKNOWN")
    selected_candidate = selection_data.get("selected_candidate")
    selection_reason = selection_data.get("selection_reason", "No selection decision available.")
    gen_time = selection_data.get("generated_at", "N/A")

    md = []
    md.append(f"# {project_name} Configuration Selection & Benchmark Evidence Report\n")
    md.append(f"**Generated at:** `{gen_time}`  ")
    md.append(f"**Track:** CraftifAI Physical AI Hackathon — Track A: PipeGen  ")
    md.append(f"**Use Case:** `{use_case}`  ")
    md.append(f"**Input Video:** `{input_video}`  ")
    md.append(f"**Overall SLO Status:** **`{overall_status}`**\n")

    md.append("## 1. Service Level Objectives (SLO)\n")
    md.append("| Metric | Target Constraint | Description |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Minimum FPS** | `≥ {slo.get('min_fps', 15)} FPS` | Throughput target for real-time video stream ingestion |")
    md.append(f"| **Maximum p95 Latency** | `≤ {slo.get('max_p95_latency_ms', 250)} ms` | Upper bound for tail frame processing latency |")
    md.append(f"| **Minimum Detection Confidence** | `≥ {slo.get('min_detection_confidence', 0.45)}` | Minimum acceptable mean detection confidence score |\n")

    md.append("## 2. Benchmark Evaluation Summary\n")
    if not bench_df.empty:
        md.append("| Candidate | FPS | p50 Latency (ms) | p95 Latency (ms) | Mean Conf | Detections | GPU Util (%) | CPU Util (%) | Status |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, row in bench_df.iterrows():
            cid = row.get("candidate_id", "unknown")
            fps = f"{row.get('fps', 0):.2f}" if pd.notnull(row.get("fps")) else "N/A"
            p50 = f"{row.get('p50_latency_ms', 0):.2f}" if pd.notnull(row.get("p50_latency_ms")) else "N/A"
            p95 = f"{row.get('p95_latency_ms', 0):.2f}" if pd.notnull(row.get("p95_latency_ms")) else "N/A"
            conf = f"{row.get('mean_confidence', 0):.4f}" if pd.notnull(row.get("mean_confidence")) else "N/A"
            det = f"{int(row.get('detection_count', 0))}" if pd.notnull(row.get("detection_count")) else "N/A"
            gpu = f"{row.get('gpu_utilization_percent')}%" if pd.notnull(row.get("gpu_utilization_percent")) and str(row.get("gpu_utilization_percent")).strip() != "" else "null"
            cpu = f"{row.get('cpu_utilization_percent')}%" if pd.notnull(row.get("cpu_utilization_percent")) and str(row.get("cpu_utilization_percent")).strip() != "" else "null"
            st = row.get("status", "unknown")
            is_sel = "**" if selected_candidate and selected_candidate.get("id") == cid else ""
            md.append(f"| {is_sel}{cid}{is_sel} | {fps} | {p50} | {p95} | {conf} | {det} | {gpu} | {cpu} | `{st}` |")
    else:
        md.append("_No benchmark results recorded in benchmark_results.csv._")
    md.append("\n")

    md.append("## 3. Configuration Selection Decision\n")
    if selected_candidate:
        md.append(f"### Selected Configuration: **`{selected_candidate['id']}`**")
        md.append(f"- **Inference Interval:** `{selected_candidate.get('inference_interval')}`")
        md.append(f"- **Resolution:** `{selected_candidate.get('resolution')}`")
        md.append(f"- **ROI Mode:** `{selected_candidate.get('roi_mode')}`")
        md.append(f"- **SLO Compliance:** `{selected_candidate.get('slo_status')}`")
        md.append(f"- **Normalized Violation Score (V_c):** `{selected_candidate.get('violation_score', 0.0):.4f}`\n")
        md.append(f"> **Technical Selection Rationale:**  \n> {selection_reason}\n")
    else:
        md.append("> **Selection Result:** No candidate was selected.\n")

    md.append("## 4. Runtime Adaptation & Guardrail Log\n")
    if adaptation_events:
        md.append("| Timestamp | Event Type | Prev Interval | New Interval | Details |")
        md.append("| :--- | :--- | :---: | :---: | :--- |")
        for ev in adaptation_events:
            ts = ev.get("timestamp", "")
            evt = ev.get("event", "")
            prev_i = ev.get("previous_interval", "-")
            new_i = ev.get("new_interval", ev.get("reverted_interval", "-"))
            reason = ev.get("reason", "")
            md.append(f"| `{ts}` | `{evt}` | {prev_i} | {new_i} | {reason} |")
    else:
        md.append("_No runtime adaptation events triggered during steady-state processing._")
    md.append("\n")

    md.append("## 5. Architectural Boundaries & Limitations\n")
    md.append("- **Single Model Scope:** The MVP evaluates one ONNX person detection model across resolution and interval variants.")
    md.append("- **Hysteresis Thresholds:** Degradation triggers after 3 consecutive latency violations; recovery requires 5 consecutive low-load windows.")
    md.append("- **Quality Preservation Guardrail:** If mean confidence drops > 20% following a downgrade event, the system reverts immediately to protect perception accuracy.")
    md.append("- **Telemetry Discipline:** Per NFR-04 and NFR-05, all metrics reflect actual hardware counters; unavailable GPU metrics are recorded as null without fabrication.\n")

    md.append("## 6. Reproduction Commands\n")
    md.append("```powershell\n# 1. Run full benchmark suite\npython -m adapipe.main --mode benchmark\n\n# 2. Run selection engine\npython -m adapipe.main --mode select\n\n# 3. Run adaptive execution on high-load video\npython -m adapipe.main --mode adaptive\n\n# 4. Generate evidence report\npython -m adapipe.main --mode report\n```\n")

    report_content = "\n".join(md)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    return out_md
