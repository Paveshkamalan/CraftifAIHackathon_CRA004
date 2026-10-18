"""
Unit tests for adapipe/validator.py
Implements test scenarios T-V01, T-V02, T-V03 from the AdaPipe Test Plan.
"""

import pytest
from pathlib import Path
from adapipe.validator import (
    validate_candidates,
    validate_single_candidate,
    ValidationError,
)


@pytest.fixture
def base_slo_config(tmp_path: Path):
    video = tmp_path / "mock.mp4"
    video.write_bytes(b"dummy video data")
    return {
        "use_case": {"input_video": str(video)},
        "slo": {
            "min_fps": 15,
            "max_p95_latency_ms": 250,
            "min_detection_confidence": 0.45,
        },
    }


def test_t_v01_missing_video_file(tmp_path: Path):
    """T-V01: Missing video file causes validation failure."""
    slo_cfg = {
        "use_case": {"input_video": str(tmp_path / "does_not_exist.mp4")},
        "slo": {"min_fps": 15, "max_p95_latency_ms": 250, "min_detection_confidence": 0.45},
    }
    candidates = [{"id": "baseline", "inference_interval": 1, "resolution": "native", "roi_mode": "full_frame"}]
    with pytest.raises(ValidationError, match="Input video validation failed"):
        validate_candidates(candidates, slo_cfg, check_artifacts=False, check_video=True)


def test_t_v02_invalid_inference_interval():
    """T-V02: Candidate with negative or invalid inference interval is rejected."""
    bad_candidate = {
        "id": "broken",
        "inference_interval": -1,
        "resolution": "native",
        "roi_mode": "full_frame",
    }
    ok, reason = validate_single_candidate(bad_candidate, check_artifacts=False)
    assert not ok
    assert "Inference interval must be integer between 1 and 10" in reason


def test_t_v03_valid_candidate():
    """T-V03: Valid candidate is accepted."""
    good_candidate = {
        "id": "balanced",
        "inference_interval": 2,
        "resolution": [960, 540],
        "roi_mode": "full_frame",
    }
    ok, reason = validate_single_candidate(good_candidate, check_artifacts=False)
    assert ok
    assert reason == ""


def test_invalid_resolution_format():
    """Invalid resolution format rejected."""
    cand = {"id": "test", "inference_interval": 1, "resolution": "invalid_res", "roi_mode": "full_frame"}
    ok, reason = validate_single_candidate(cand, check_artifacts=False)
    assert not ok
    assert "Resolution must be 'native' or [width, height]" in reason


def test_invalid_roi_mode():
    """Invalid ROI mode rejected."""
    cand = {"id": "test", "inference_interval": 1, "resolution": "native", "roi_mode": "circular_roi"}
    ok, reason = validate_single_candidate(cand, check_artifacts=False)
    assert not ok
    assert "ROI mode must be 'full_frame' or 'activity_roi'" in reason


def test_missing_artifact_rejection(tmp_path: Path):
    """Candidate with missing pipegen_config_path is rejected when check_artifacts=True."""
    cand = {
        "id": "baseline",
        "inference_interval": 1,
        "resolution": "native",
        "roi_mode": "full_frame",
        "pipegen_config_path": str(tmp_path / "missing_config.yaml"),
    }
    ok, reason = validate_single_candidate(cand, check_artifacts=True)
    assert not ok
    assert "Required PipeGen config artifact does not exist" in reason
