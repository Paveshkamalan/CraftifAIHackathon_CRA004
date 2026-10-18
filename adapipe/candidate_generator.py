"""
Candidate generator for AdaPipe.
Reads candidate configurations and produces the bounded candidate space.
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml


class CandidateGenerationError(Exception):
    """Raised when candidate generation encounters invalid configuration."""
    pass


def generate_candidates(candidates_path: str | Path = "configs/candidates.yaml") -> List[Dict[str, Any]]:
    """
    Generate exactly three candidate configurations for MVP based on candidates.yaml.

    Args:
        candidates_path: Path to candidates.yaml.

    Returns:
        List of candidate dictionaries conforming to the AdaPipe candidate data model.

    Raises:
        CandidateGenerationError: If file is missing, empty, or candidate count != 3.
    """
    path = Path(candidates_path)
    if not path.is_file():
        raise CandidateGenerationError(f"Candidate configuration file does not exist: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise CandidateGenerationError(f"Error parsing candidates YAML: {exc}") from exc

    if not isinstance(raw, dict) or "candidates" not in raw:
        raise CandidateGenerationError("Missing 'candidates' root list in configuration.")

    raw_candidates = raw["candidates"]
    if not isinstance(raw_candidates, list):
        raise CandidateGenerationError("'candidates' must be a list of candidate configurations.")

    if len(raw_candidates) != 3:
        raise CandidateGenerationError(
            f"AdaPipe MVP requires exactly 3 candidate configurations. Found {len(raw_candidates)}."
        )

    candidates: List[Dict[str, Any]] = []
    for c in raw_candidates:
        if not isinstance(c, dict):
            raise CandidateGenerationError("Each candidate definition must be a dictionary.")

        cand_id = c.get("id")
        if not cand_id:
            raise CandidateGenerationError("Candidate missing required 'id'.")

        cand_obj: Dict[str, Any] = {
            "id": str(cand_id),
            "inference_interval": c.get("inference_interval", 1),
            "resolution": c.get("resolution", "native"),
            "roi_mode": c.get("roi_mode", "full_frame"),
            "pipegen_config_path": c.get("pipegen_config_path", "pipegen_output/generated_pipeline/config.yaml"),
            "status": "pending",
            "slo_status": None,
            "metrics": {},
        }
        candidates.append(cand_obj)

    return candidates
