"""
Unit tests for adapipe/selector.py
Implements test scenarios T-S01, T-S02, T-S03 from the AdaPipe Test Plan.
"""

import pytest
from adapipe.selector import (
    evaluate_candidate_slo,
    select_best_configuration,
    create_selection_report,
)


@pytest.fixture
def slo_config():
    return {
        "project": {"name": "AdaPipe"},
        "use_case": {"name": "person_detection_entry", "input_video": "demo/sample_video.mp4"},
        "slo": {
            "min_fps": 15.0,
            "max_p95_latency_ms": 250.0,
            "min_detection_confidence": 0.45,
        },
    }


def test_t_s01_one_feasible_candidate(slo_config):
    """T-S01: When exactly one candidate meets the SLO, it is selected."""
    candidates = [
        {
            "id": "c1",
            "status": "success",
            "inference_interval": 1,
            "metrics": {"fps": 25.0, "p95_latency_ms": 120.0, "mean_confidence": 0.60},
        },
        {
            "id": "c2",
            "status": "success",
            "inference_interval": 2,
            "metrics": {"fps": 10.0, "p95_latency_ms": 300.0, "mean_confidence": 0.50},  # fails FPS & latency
        },
    ]

    selected, overall_status, reason, evaluated = select_best_configuration(candidates, slo_config)
    assert selected is not None
    assert selected["id"] == "c1"
    assert overall_status == "SLO_MET"
    assert "Candidate 'c1' meets all SLO requirements" in reason


def test_t_s02_multiple_feasible_candidates_highest_confidence(slo_config):
    """T-S02: When multiple candidates pass, select the one with highest mean confidence."""
    candidates = [
        {
            "id": "baseline",
            "status": "success",
            "inference_interval": 1,
            "metrics": {"fps": 20.0, "p95_latency_ms": 150.0, "mean_confidence": 0.85},
        },
        {
            "id": "balanced",
            "status": "success",
            "inference_interval": 2,
            "metrics": {"fps": 35.0, "p95_latency_ms": 80.0, "mean_confidence": 0.75},
        },
        {
            "id": "fast",
            "status": "success",
            "inference_interval": 3,
            "metrics": {"fps": 50.0, "p95_latency_ms": 40.0, "mean_confidence": 0.65},
        },
    ]

    selected, overall_status, reason, evaluated = select_best_configuration(candidates, slo_config)
    assert selected is not None
    assert selected["id"] == "baseline"
    assert overall_status == "SLO_MET"
    assert selected["metrics"]["mean_confidence"] == 0.85


def test_t_s03_no_feasible_candidate_selects_least_violating(slo_config):
    """T-S03: When no candidate meets SLO, choose least-violating candidate and set SLO_UNMET."""
    candidates = [
        {
            "id": "bad1",
            "status": "success",
            "inference_interval": 1,
            "metrics": {"fps": 14.0, "p95_latency_ms": 260.0, "mean_confidence": 0.44},  # slight violation
        },
        {
            "id": "bad2",
            "status": "success",
            "inference_interval": 2,
            "metrics": {"fps": 5.0, "p95_latency_ms": 600.0, "mean_confidence": 0.20},  # massive violation
        },
    ]

    selected, overall_status, reason, evaluated = select_best_configuration(candidates, slo_config)
    assert selected is not None
    assert selected["id"] == "bad1"
    assert overall_status == "SLO_UNMET"
    assert "No candidate met all SLO constraints" in reason
    assert selected["violation_score"] < candidates[1]["violation_score"]


def test_create_selection_report_structure(slo_config):
    """Verify selection report matches JSON schema required by DOCUMENT 6."""
    candidates = [
        {
            "id": "balanced",
            "status": "success",
            "inference_interval": 2,
            "metrics": {"fps": 25.0, "p95_latency_ms": 100.0, "mean_confidence": 0.70},
        }
    ]
    report = create_selection_report(candidates, slo_config)
    assert report["project"] == "AdaPipe"
    assert report["use_case"] == "person_detection_entry"
    assert report["overall_slo_status"] == "SLO_MET"
    assert report["selected_candidate"]["id"] == "balanced"
    assert len(report["limitations"]) > 0
    assert "generated_at" in report
