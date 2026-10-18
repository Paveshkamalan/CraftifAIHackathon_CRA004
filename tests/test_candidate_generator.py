"""
Unit tests for adapipe/candidate_generator.py
Ensures candidate generation strictly adheres to MVP constraints.
"""

import pytest
import yaml
from pathlib import Path
from adapipe.candidate_generator import generate_candidates, CandidateGenerationError


def test_generate_exactly_three_candidates():
    """Default candidates.yaml produces exactly three valid candidate dictionaries."""
    candidates = generate_candidates("configs/candidates.yaml")
    assert len(candidates) == 3
    ids = [c["id"] for c in candidates]
    assert ids == ["baseline", "balanced", "fast"]

    baseline = candidates[0]
    assert baseline["inference_interval"] == 1
    assert baseline["resolution"] == "native"
    assert baseline["roi_mode"] == "full_frame"
    assert baseline["status"] == "pending"

    balanced = candidates[1]
    assert balanced["inference_interval"] == 2
    assert balanced["resolution"] == [960, 540]

    fast = candidates[2]
    assert fast["inference_interval"] == 3
    assert fast["resolution"] == [640, 360]


def test_reject_more_or_less_than_three_candidates(tmp_path: Path):
    """Configuration with not exactly 3 candidates raises CandidateGenerationError."""
    bad_cfg = {
        "candidates": [
            {"id": "c1", "inference_interval": 1},
            {"id": "c2", "inference_interval": 2},
        ]
    }
    path = tmp_path / "bad_candidates.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(bad_cfg, f)

    with pytest.raises(CandidateGenerationError, match="AdaPipe MVP requires exactly 3 candidate configurations"):
        generate_candidates(path)
