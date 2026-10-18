"""
AdaPipe Main Orchestrator.
Entry point for Track A: PipeGen constraint-aware control plane.
Runs configuration loading, candidate generation, validation,
benchmarking, SLO selection, runtime adaptation, and evidence reporting.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from adapipe.config_loader import load_config, ConfigurationError
from adapipe.candidate_generator import generate_candidates, CandidateGenerationError
from adapipe.validator import validate_candidates, ValidationError
from adapipe.benchmark_runner import run_benchmarks
from adapipe.selector import (
    create_selection_report,
    save_selection_report,
    select_best_configuration,
)
from adapipe.runtime_policy import run_adaptive_pipeline
from adapipe.report import generate_markdown_report
from adapipe.ble_event_source import SimulatedBLEEventSource


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdaPipe: Constraint-Aware, Benchmark-Driven Control Layer for PipeGen Pipelines."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/slo.yaml",
        help="Path to SLO configuration YAML file (default: configs/slo.yaml)",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="configs/candidates.yaml",
        help="Path to candidates YAML file (default: configs/candidates.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "benchmark", "select", "adaptive", "report", "ble"],
        default="all",
        help="Operational mode: all, benchmark, select, adaptive, report, ble (default: all)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration override in seconds for benchmarking",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=None,
        help="Optional warmup override in seconds for benchmarking",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional input video path override",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=================================================================")
    print("  AdaPipe: Pipelines that meet the deadline — not just run.")
    print("  Track A: PipeGen | CraftifAI Physical AI Hackathon")
    print("=================================================================")

    # 1. Load Configuration
    print(f"\n[Phase 1] Loading configuration from {args.config}...")
    try:
        slo_cfg = load_config(args.config)
        if args.video:
            slo_cfg["use_case"]["input_video"] = args.video
        print(f"  Project: {slo_cfg['project']['name']} v{slo_cfg['project'].get('version', '1.0.0')}")
        print(f"  Use Case: {slo_cfg['use_case']['name']}")
        print(f"  SLO: min_fps={slo_cfg['slo']['min_fps']}, max_p95={slo_cfg['slo']['max_p95_latency_ms']}ms, min_conf={slo_cfg['slo']['min_detection_confidence']}")
    except ConfigurationError as err:
        print(f"FATAL: Configuration Error: {err}", file=sys.stderr)
        return 1

    # 2. Candidate Generation & Validation
    print(f"\n[Phase 2] Generating and validating candidates from {args.candidates}...")
    try:
        raw_candidates = generate_candidates(args.candidates)
        valid_candidates, invalid_candidates = validate_candidates(
            raw_candidates, slo_cfg, check_artifacts=True, check_video=True
        )
        print(f"  Total Candidates: {len(raw_candidates)}")
        print(f"  Valid Candidates: {len(valid_candidates)}")
        if invalid_candidates:
            print(f"  Invalid Candidates: {len(invalid_candidates)}")
            for inv in invalid_candidates:
                print(f"    - {inv['id']}: {inv.get('rejection_reason')}")
    except (CandidateGenerationError, ValidationError) as err:
        print(f"FATAL: Validation Error: {err}", file=sys.stderr)
        return 1

    if args.mode in ["all", "benchmark"]:
        print(f"\n[Phase 3 & 4] Executing Sequential Benchmarks on PipeGen Pipeline...")
        duration = args.duration if args.duration is not None else slo_cfg.get("benchmark", {}).get("duration_seconds", 45)
        warmup = args.warmup if args.warmup is not None else slo_cfg.get("benchmark", {}).get("warmup_seconds", 5)
        print(f"  Benchmark duration: {duration}s (warmup: {warmup}s)")

        benchmark_results = run_benchmarks(
            valid_candidates + invalid_candidates,
            slo_cfg,
            duration_override=duration,
            warmup_override=warmup,
        )
        csv_path = slo_cfg.get("output", {}).get("benchmark_csv", "results/benchmark_results.csv")
        print(f"  Wrote {len(benchmark_results)} benchmark rows to {csv_path}")

    if args.mode in ["all", "select"]:
        print(f"\n[Phase 5] Running SLO Evaluation & Configuration Selection...")
        selection_report = create_selection_report(valid_candidates, slo_cfg)
        json_path = slo_cfg.get("output", {}).get("selection_json", "results/selection_report.json")
        save_selection_report(selection_report, json_path)

        sel = selection_report.get("selected_candidate")
        if sel:
            print(f"  >>> SELECTED CONFIGURATION: '{sel['id']}' <<<")
            print(f"  SLO Status: {selection_report['overall_slo_status']}")
            print(f"  Rationale: {selection_report['selection_reason']}")
        else:
            print("  WARNING: No candidate could be selected.")
        print(f"  Wrote selection report to {json_path}")

    if args.mode in ["all", "adaptive"]:
        print(f"\n[Phase 6] Running Live Runtime Adaptation Under Workload...")
        high_load_video = slo_cfg.get("use_case", {}).get("high_load_video", "demo/high_load_video.mp4")

        # Pick selected candidate or first valid candidate
        selected_cand = valid_candidates[0]
        json_path = slo_cfg.get("output", {}).get("selection_json", "results/selection_report.json")
        if Path(json_path).exists():
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                sel_data = json.load(f)
                if sel_data.get("selected_candidate"):
                    cand_id = sel_data["selected_candidate"]["id"]
                    match = [c for c in valid_candidates if c["id"] == cand_id]
                    if match:
                        selected_cand = match[0]

        adapt_res = run_adaptive_pipeline(
            candidate=selected_cand,
            video_path=high_load_video,
            slo_config=slo_cfg,
            duration_seconds=args.duration or 10.0,
            warmup_seconds=args.warmup or 1.0,
            synthetic_high_load_stress=True,
        )
        print(f"  Adaptive Execution Completed:")
        print(f"  Initial Interval: {selected_cand.get('inference_interval', 1)} -> Final Interval: {adapt_res['final_interval']}")
        print(f"  Adaptation Events Triggered: {len(adapt_res['adaptation_events'])}")
        print(f"  Events Logged to: {adapt_res['log_file']}")

    if args.mode in ["all", "report"]:
        print(f"\n[Phase 7] Generating Markdown Evidence Report...")
        md_file = generate_markdown_report(
            selection_report_path=slo_cfg.get("output", {}).get("selection_json", "results/selection_report.json"),
            benchmark_csv_path=slo_cfg.get("output", {}).get("benchmark_csv", "results/benchmark_results.csv"),
            adaptation_log_path=slo_cfg.get("output", {}).get("adaptation_log", "results/adaptation_log.jsonl"),
            output_md_path=slo_cfg.get("output", {}).get("report_md", "results/selection_report.md"),
        )
        print(f"  Report generated at: {md_file}")

    if args.mode == "ble":
        print(f"\n[Demo Mode] Publishing Simulated ESP32 BLE Advertisements...")
        ble_source = SimulatedBLEEventSource()
        ble_source.listen(lambda evt: print(f"  [BLE RECV] {evt['device_id']} ({evt['zone']}): {evt['event_type']} | RSSI={evt['rssi_dbm']}dBm, Bat={evt['battery_mv']}mV"), max_events=5)

    print("\n=================================================================")
    print("  AdaPipe Execution Complete. Artifacts Ready.")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
