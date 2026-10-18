"""
Integration tests for AdaPipe.
Implements I-01 through I-05 from the AdaPipe Test Plan:
- I-01: Run full benchmark on sample video -> CSV contains valid rows
- I-02: Run selection after benchmark -> JSON report generated
- I-03: Run high-load video -> Adaptation event appears
- I-04: Run low-load recovery -> Upgrade event appears
- I-05: Missing PipeGen output -> Clear failure message
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from adapipe.config_loader import load_config
from adapipe.candidate_generator import generate_candidates
from adapipe.validator import validate_candidates, validate_single_candidate
from adapipe.benchmark_runner import run_benchmarks
from adapipe.selector import create_selection_report, save_selection_report
from adapipe.runtime_policy import run_adaptive_pipeline, RuntimePolicy


@pytest.fixture
def test_config():
    return load_config("configs/slo.yaml")


def test_i_01_full_benchmark_produces_csv(test_config, tmp_path: Path):
    """I-01: Run benchmark suite on sample video; CSV contains valid rows."""
    candidates = generate_candidates("configs/candidates.yaml")
    val_cands, _ = validate_candidates(candidates, test_config)

    test_csv = tmp_path / "benchmark_results.csv"
    test_config["output"]["benchmark_csv"] = str(test_csv)

    results = run_benchmarks(
        val_cands,
        test_config,
        duration_override=1.5,
        warmup_override=0.5,
    )

    assert len(results) == 3
    assert test_csv.exists()
    df = pd.read_csv(test_csv)
    assert len(df) == 3
    assert "baseline" in df["candidate_id"].values
    assert "balanced" in df["candidate_id"].values
    assert "fast" in df["candidate_id"].values
    for _, row in df.iterrows():
        assert row["status"] == "success"
        assert row["fps"] > 0
        assert row["p95_latency_ms"] > 0


def test_i_02_run_selection_generates_json(test_config, tmp_path: Path):
    """I-02: Run selection after benchmark; valid selection_report.json is generated."""
    candidates = generate_candidates("configs/candidates.yaml")
    # Simulate benchmark metrics
    candidates[0]["status"] = "success"
    candidates[0]["metrics"] = {"fps": 18.0, "p95_latency_ms": 120.0, "mean_confidence": 0.72}
    candidates[1]["status"] = "success"
    candidates[1]["metrics"] = {"fps": 30.0, "p95_latency_ms": 60.0, "mean_confidence": 0.68}
    candidates[2]["status"] = "success"
    candidates[2]["metrics"] = {"fps": 45.0, "p95_latency_ms": 30.0, "mean_confidence": 0.65}

    report = create_selection_report(candidates, test_config)
    json_path = tmp_path / "selection_report.json"
    save_selection_report(report, json_path)

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["overall_slo_status"] == "SLO_MET"
    assert saved["selected_candidate"]["id"] == "baseline"  # highest confidence among feasible
    assert len(saved["candidates_evaluated"]) == 3


def test_i_03_high_load_triggers_adaptation(test_config, tmp_path: Path):
    """I-03: Run high-load video; adaptation degradation event appears."""
    log_path = tmp_path / "adaptation_log.jsonl"
    test_config["output"]["adaptation_log"] = str(log_path)

    candidate = {"id": "baseline", "inference_interval": 1, "resolution": "native", "roi_mode": "full_frame"}

    res = run_adaptive_pipeline(
        candidate=candidate,
        video_path="demo/high_load_video.mp4",
        slo_config=test_config,
        duration_seconds=3.0,
        warmup_seconds=0.5,
        synthetic_high_load_stress=True,
    )

    assert res["final_interval"] > 1
    assert len(res["adaptation_events"]) > 0
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    evt = json.loads(lines[0])
    assert evt["event"] == "degrade_configuration"


def test_i_04_low_load_triggers_recovery(tmp_path: Path):
    """I-04: Sustained low latency triggers recovery upgrade event."""
    log_file = tmp_path / "recovery.jsonl"
    policy = RuntimePolicy(
        initial_interval=1,
        max_p95_latency_ms=250.0,
        recover_after_success_windows=3,
        recovery_latency_ratio=0.70,
        log_file=log_file,
    )
    policy.current_interval = 2  # starts degraded

    events = []
    for _ in range(3):
        ev = policy.process_window(window_p95=80.0, window_mean_conf=0.75)
        if ev:
            events.append(ev)

    assert len(events) == 1
    assert events[0]["event"] == "recover_configuration"
    assert policy.current_interval == 1


def test_i_05_missing_pipegen_output(tmp_path: Path):
    """I-05: Missing PipeGen artifact rejected cleanly with descriptive error."""
    cand = {
        "id": "missing_artifact",
        "inference_interval": 1,
        "resolution": "native",
        "roi_mode": "full_frame",
        "pipegen_config_path": str(tmp_path / "non_existent_pipegen.yaml"),
    }
    is_valid, reason = validate_single_candidate(cand, check_artifacts=True)
    assert not is_valid
    assert "Required PipeGen config artifact does not exist" in reason
