"""
Unit tests for adapipe/config_loader.py
Implements test scenarios T-C01, T-C02, T-C03 from the AdaPipe Test Plan.
"""

import pytest
import yaml
from pathlib import Path
from adapipe.config_loader import load_config, ConfigurationError


def test_t_c01_valid_yaml(tmp_path: Path):
    """T-C01: Valid YAML configuration loads cleanly."""
    valid_content = {
        "project": {"name": "AdaPipe", "version": "1.0.0", "track": "Track_A"},
        "use_case": {
            "name": "person_detection_entry",
            "description": "Detect people",
            "input_video": "demo/sample_video.mp4",
        },
        "slo": {
            "min_fps": 15,
            "max_p95_latency_ms": 250,
            "min_detection_confidence": 0.45,
        },
        "benchmark": {
            "duration_seconds": 45,
            "warmup_seconds": 5,
            "metric_window_seconds": 5,
        },
        "runtime_policy": {
            "degrade_after_violations": 3,
            "recover_after_success_windows": 5,
            "recovery_latency_ratio": 0.7,
            "quality_drop_revert_threshold": 0.20,
        },
        "output": {
            "benchmark_csv": "results/benchmark_results.csv",
            "selection_json": "results/selection_report.json",
        },
    }
    cfg_file = tmp_path / "slo.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(valid_content, f)

    loaded = load_config(cfg_file)
    assert loaded["project"]["name"] == "AdaPipe"
    assert loaded["slo"]["min_fps"] == 15
    assert loaded["slo"]["max_p95_latency_ms"] == 250
    assert loaded["slo"]["min_detection_confidence"] == 0.45


def test_t_c02_negative_fps(tmp_path: Path):
    """T-C02: Negative or zero FPS raises ConfigurationError."""
    bad_content = {
        "project": {"name": "AdaPipe"},
        "use_case": {"name": "test", "input_video": "demo/sample.mp4"},
        "slo": {
            "min_fps": -5,
            "max_p95_latency_ms": 250,
            "min_detection_confidence": 0.45,
        },
    }
    cfg_file = tmp_path / "slo_bad_fps.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(bad_content, f)

    with pytest.raises(ConfigurationError, match="min_fps.*positive"):
        load_config(cfg_file)


def test_t_c03_missing_input_video(tmp_path: Path):
    """T-C03: Missing input video path raises ConfigurationError."""
    bad_content = {
        "project": {"name": "AdaPipe"},
        "use_case": {"name": "test"},  # missing input_video
        "slo": {
            "min_fps": 15,
            "max_p95_latency_ms": 250,
            "min_detection_confidence": 0.45,
        },
    }
    cfg_file = tmp_path / "slo_missing_video.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(bad_content, f)

    with pytest.raises(ConfigurationError, match="input_video"):
        load_config(cfg_file)


def test_missing_file_raises_error():
    """Non-existent file raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config("non_existent_file.yaml")


def test_invalid_confidence_range(tmp_path: Path):
    """Confidence outside 0..1 raises ConfigurationError."""
    bad_content = {
        "project": {"name": "AdaPipe"},
        "use_case": {"name": "test", "input_video": "demo/sample.mp4"},
        "slo": {
            "min_fps": 15,
            "max_p95_latency_ms": 250,
            "min_detection_confidence": 1.5,
        },
    }
    cfg_file = tmp_path / "slo_bad_conf.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(bad_content, f)

    with pytest.raises(ConfigurationError, match="min_detection_confidence.*between 0.0 and 1.0"):
        load_config(cfg_file)
