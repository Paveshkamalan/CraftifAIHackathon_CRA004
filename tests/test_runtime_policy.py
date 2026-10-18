"""
Unit tests for adapipe/runtime_policy.py
Implements test scenarios T-P01, T-P02, T-P03 from the AdaPipe Test Plan.
"""

from pathlib import Path
import pytest
from adapipe.runtime_policy import RuntimePolicy


def test_t_p01_degrade_after_three_violations(tmp_path: Path):
    """T-P01: Three consecutive windows exceeding SLO latency triggers degradation."""
    log_file = tmp_path / "adaptation.jsonl"
    policy = RuntimePolicy(
        initial_interval=1,
        max_p95_latency_ms=250.0,
        degrade_after_violations=3,
        recover_after_success_windows=5,
        log_file=log_file,
    )

    # Window 1 violation (> 250ms)
    ev1 = policy.process_window(window_p95=280.0, window_mean_conf=0.75)
    assert ev1 is None
    assert policy.current_interval == 1
    assert policy.consecutive_violations == 1

    # Window 2 violation
    ev2 = policy.process_window(window_p95=290.0, window_mean_conf=0.75)
    assert ev2 is None
    assert policy.current_interval == 1
    assert policy.consecutive_violations == 2

    # Window 3 violation -> Degrade!
    ev3 = policy.process_window(window_p95=310.0, window_mean_conf=0.74)
    assert ev3 is not None
    assert ev3["event"] == "degrade_configuration"
    assert ev3["previous_interval"] == 1
    assert ev3["new_interval"] == 2
    assert policy.current_interval == 2

    # Verify written to log_file
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "degrade_configuration" in lines[0]


def test_t_p02_recover_after_five_low_latency_windows(tmp_path: Path):
    """T-P02: Five consecutive windows below 70% of SLO latency triggers recovery."""
    log_file = tmp_path / "adaptation.jsonl"
    # Initial was 1, currently at 2 (degraded)
    policy = RuntimePolicy(
        initial_interval=1,
        max_p95_latency_ms=250.0,
        recover_after_success_windows=5,
        recovery_latency_ratio=0.70,  # 0.7 * 250 = 175ms
        log_file=log_file,
    )
    policy.current_interval = 2

    # Windows 1 to 4 with low latency (< 175ms)
    for i in range(4):
        ev = policy.process_window(window_p95=120.0, window_mean_conf=0.70)
        assert ev is None
        assert policy.current_interval == 2

    # Window 5 with low latency -> Upgrade!
    ev5 = policy.process_window(window_p95=115.0, window_mean_conf=0.70)
    assert ev5 is not None
    assert ev5["event"] == "recover_configuration"
    assert ev5["previous_interval"] == 2
    assert ev5["new_interval"] == 1
    assert policy.current_interval == 1


def test_t_p03_quality_guardrail_reverts(tmp_path: Path):
    """T-P03: Confidence drop > 20% right after degradation triggers quality guardrail revert."""
    log_file = tmp_path / "adaptation.jsonl"
    policy = RuntimePolicy(
        initial_interval=1,
        max_p95_latency_ms=250.0,
        degrade_after_violations=1,  # immediate for testing
        quality_drop_revert_threshold=0.20,
        log_file=log_file,
    )

    # 1. Trigger degrade from 1 -> 2 with baseline confidence 0.80
    ev_deg = policy.process_window(window_p95=300.0, window_mean_conf=0.80)
    assert ev_deg is not None
    assert ev_deg["event"] == "degrade_configuration"
    assert policy.current_interval == 2

    # 2. Next window confidence drops drastically to 0.50 (> 20% drop from 0.80: (0.80-0.50)/0.80 = 37.5%)
    ev_guard = policy.process_window(window_p95=150.0, window_mean_conf=0.50)
    assert ev_guard is not None
    assert ev_guard["event"] == "quality_guardrail_triggered"
    assert ev_guard["previous_interval"] == 2
    assert ev_guard["reverted_interval"] == 1
    assert policy.current_interval == 1


def test_no_upgrade_beyond_initial_selected():
    """MVP constraint: Pipeline shall not upgrade beyond the original selected configuration."""
    policy = RuntimePolicy(
        initial_interval=2,  # original selected configuration was 2
        max_p95_latency_ms=250.0,
        recover_after_success_windows=2,
    )
    # Five consecutive low latency windows
    for _ in range(5):
        policy.process_window(window_p95=50.0, window_mean_conf=0.80)

    # Should remain at initial_interval 2, not upgrade to 1
    assert policy.current_interval == 2
