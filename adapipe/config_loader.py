"""
Configuration loader for AdaPipe.
Ingests and validates SLO, use case, benchmark, and runtime policy settings.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigurationError(Exception):
    """Raised when an AdaPipe configuration fails validation."""
    pass


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load and validate the SLO YAML configuration file.

    Args:
        config_path: Path to the slo.yaml file.

    Returns:
        Validated configuration dictionary.

    Raises:
        ConfigurationError: If the file is missing, invalid YAML, or fails schema validation.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML parsing error in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError("Root configuration must be a dictionary/mapping.")

    validate_slo_config(data)
    return data


def validate_slo_config(cfg: Dict[str, Any]) -> None:
    """
    Validate the structure and constraints of the AdaPipe SLO configuration.
    """
    # 1. Project
    project = cfg.get("project")
    if not isinstance(project, dict):
        raise ConfigurationError("Missing or invalid 'project' section.")
    if not project.get("name"):
        raise ConfigurationError("Project 'name' is required.")

    # 2. Use Case
    use_case = cfg.get("use_case")
    if not isinstance(use_case, dict):
        raise ConfigurationError("Missing or invalid 'use_case' section.")
    if not use_case.get("name"):
        raise ConfigurationError("Use case 'name' is required.")
    input_video = use_case.get("input_video")
    if not input_video or not isinstance(input_video, str):
        raise ConfigurationError("Missing or invalid 'input_video' path in use_case.")

    # 3. SLO
    slo = cfg.get("slo")
    if not isinstance(slo, dict):
        raise ConfigurationError("Missing or invalid 'slo' section.")

    min_fps = slo.get("min_fps")
    if min_fps is None or not isinstance(min_fps, (int, float)) or min_fps <= 0:
        raise ConfigurationError(f"SLO 'min_fps' must be a positive number. Got: {min_fps}")

    max_p95_latency_ms = slo.get("max_p95_latency_ms")
    if max_p95_latency_ms is None or not isinstance(max_p95_latency_ms, (int, float)) or max_p95_latency_ms <= 0:
        raise ConfigurationError(
            f"SLO 'max_p95_latency_ms' must be a positive number. Got: {max_p95_latency_ms}"
        )

    min_detection_confidence = slo.get("min_detection_confidence")
    if (
        min_detection_confidence is None
        or not isinstance(min_detection_confidence, (int, float))
        or not (0.0 <= min_detection_confidence <= 1.0)
    ):
        raise ConfigurationError(
            f"SLO 'min_detection_confidence' must be between 0.0 and 1.0. Got: {min_detection_confidence}"
        )

    # 4. Benchmark
    bench = cfg.get("benchmark", {})
    if not isinstance(bench, dict):
        raise ConfigurationError("Invalid 'benchmark' section.")
    duration = bench.get("duration_seconds", 45)
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ConfigurationError(f"Benchmark 'duration_seconds' must be positive. Got: {duration}")
    warmup = bench.get("warmup_seconds", 5)
    if not isinstance(warmup, (int, float)) or warmup < 0 or warmup >= duration:
        raise ConfigurationError(
            f"Benchmark 'warmup_seconds' must be >= 0 and less than duration_seconds. Got: {warmup}"
        )

    # 5. Runtime Policy
    rp = cfg.get("runtime_policy", {})
    if not isinstance(rp, dict):
        raise ConfigurationError("Invalid 'runtime_policy' section.")
    degrade_after = rp.get("degrade_after_violations", 3)
    if not isinstance(degrade_after, int) or degrade_after <= 0:
        raise ConfigurationError("Runtime policy 'degrade_after_violations' must be a positive integer.")
    recover_after = rp.get("recover_after_success_windows", 5)
    if not isinstance(recover_after, int) or recover_after <= 0:
        raise ConfigurationError("Runtime policy 'recover_after_success_windows' must be a positive integer.")

    # 6. Output
    output = cfg.get("output", {})
    if not isinstance(output, dict):
        raise ConfigurationError("Invalid 'output' section.")
