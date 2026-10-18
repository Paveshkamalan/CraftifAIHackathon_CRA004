"""
Benchmark runner for AdaPipe.
Orchestrates candidate benchmarking over PipeGen vision pipelines.
Enforces DOCUMENT 6 protocol: sequential execution, warmup exclusion,
clean isolation, and CSV artifact generation.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback

from pipegen_output.generated_pipeline.pipeline import PipeGenPipeline
from adapipe.metrics_collector import record_run_metrics, create_empty_metrics


CSV_FIELDNAMES = [
    "candidate_id",
    "frames_processed",
    "elapsed_seconds",
    "fps",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_confidence",
    "detection_count",
    "gpu_utilization_percent",
    "gpu_memory_used_mb",
    "cpu_utilization_percent",
    "memory_used_mb",
    "started_at",
    "ended_at",
    "status",
    "error",
]


def write_benchmark_csv(
    metrics_records: List[Dict[str, Any]],
    csv_path: str | Path = "results/benchmark_results.csv",
    append: bool = False,
) -> Path:
    """
    Write or append metrics records to benchmark_results.csv according to DOCUMENT 6 format.
    """
    out_file = Path(csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append and out_file.exists() else "w"
    write_header = (mode == "w") or (out_file.stat().st_size == 0 if out_file.exists() else True)

    with open(out_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for rec in metrics_records:
            row = {}
            for field in CSV_FIELDNAMES:
                val = rec.get(field)
                # Format None as null / empty string per CSV standards
                row[field] = "" if val is None else val
            writer.writerow(row)

    return out_file


def run_single_candidate(
    candidate: Dict[str, Any],
    video_path: str,
    duration_seconds: float,
    warmup_seconds: float,
    log_dir: Path = Path("results/logs"),
) -> Dict[str, Any]:
    """
    Execute a single candidate pipeline and collect all performance and hardware metrics.
    """
    cand_id = candidate["id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{cand_id}.log"

    started_at = datetime.now(timezone.utc)
    pipeline_res = {}
    error_msg = None

    try:
        pipeline = PipeGenPipeline(
            config_path=candidate.get("pipegen_config_path", "pipegen_output/generated_pipeline/config.yaml"),
            inference_interval=candidate.get("inference_interval", 1),
            resolution=candidate.get("resolution", "native"),
            roi_mode=candidate.get("roi_mode", "full_frame"),
        )

        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"=== Starting benchmark for candidate: {cand_id} ===\n")
            lf.write(f"Started at: {started_at.isoformat()}\n")
            lf.write(f"Parameters: interval={pipeline.inference_interval}, "
                     f"res={pipeline.resolution}, roi={pipeline.roi_mode}\n\n")

            pipeline_res = pipeline.run(
                video_path=video_path,
                duration_seconds=duration_seconds,
                warmup_seconds=warmup_seconds,
            )

            lf.write(f"Benchmark finished successfully.\n")
            lf.write(f"Frames processed: {pipeline_res['frames_processed']}\n")
            lf.write(f"FPS: {pipeline_res['fps']}\n")
            lf.write(f"p50 Latency (ms): {pipeline_res['p50_latency_ms']}\n")
            lf.write(f"p95 Latency (ms): {pipeline_res['p95_latency_ms']}\n")
            lf.write(f"Mean Confidence: {pipeline_res['mean_confidence']}\n")

    except Exception as exc:
        error_msg = str(exc)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\nERROR: Candidate execution failed:\n")
            lf.write(traceback.format_exc())

    ended_at = datetime.now(timezone.utc)
    metrics = record_run_metrics(
        candidate_id=cand_id,
        pipeline_results=pipeline_res,
        started_at=started_at,
        ended_at=ended_at,
        error=error_msg,
    )
    return metrics


def run_benchmarks(
    candidates: List[Dict[str, Any]],
    slo_config: Dict[str, Any],
    duration_override: Optional[float] = None,
    warmup_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run the sequential benchmark suite across all valid candidates.
    Appends results to CSV after each candidate run.
    """
    use_case = slo_config.get("use_case", {})
    video_path = use_case.get("input_video", "demo/sample_video.mp4")

    bench_cfg = slo_config.get("benchmark", {})
    duration = duration_override if duration_override is not None else bench_cfg.get("duration_seconds", 45)
    warmup = warmup_override if warmup_override is not None else bench_cfg.get("warmup_seconds", 5)

    csv_path = slo_config.get("output", {}).get("benchmark_csv", "results/benchmark_results.csv")

    results: List[Dict[str, Any]] = []

    # Reset CSV for fresh benchmark suite
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(csv_path).exists():
        Path(csv_path).unlink()

    for cand in candidates:
        if cand.get("status") == "invalid":
            # Record invalid candidate as failed row
            now = datetime.now(timezone.utc).isoformat()
            err_metrics = create_empty_metrics(cand["id"])
            err_metrics.update({
                "started_at": now,
                "ended_at": now,
                "status": "failed",
                "error": cand.get("rejection_reason", "Candidate rejected during validation"),
            })
            cand["metrics"] = err_metrics
            results.append(err_metrics)
            write_benchmark_csv([err_metrics], csv_path=csv_path, append=True)
            continue

        print(f"--> Running benchmark for candidate: {cand['id']} (interval={cand.get('inference_interval')}, res={cand.get('resolution')})")
        metrics = run_single_candidate(
            candidate=cand,
            video_path=video_path,
            duration_seconds=duration,
            warmup_seconds=warmup,
        )
        cand["metrics"] = metrics
        cand["status"] = metrics["status"]
        results.append(metrics)
        write_benchmark_csv([metrics], csv_path=csv_path, append=True)
        print(f"    Completed {cand['id']}: FPS={metrics['fps']}, p95={metrics['p95_latency_ms']}ms, conf={metrics['mean_confidence']}")

    return results
