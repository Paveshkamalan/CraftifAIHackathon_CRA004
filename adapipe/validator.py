"""
Candidate and environment validator for AdaPipe.
Enforces FR-03: validates input video, SLO constraints, and candidate configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2


class ValidationError(Exception):
    """Raised when environment or global prerequisites fail validation."""
    pass


def is_video_readable(video_path: str | Path) -> Tuple[bool, str]:
    """
    Check if the specified video file exists and can be decoded by OpenCV.

    Returns:
        (is_valid, error_message)
    """
    path = Path(video_path)
    if not path.is_file():
        return False, f"Video file does not exist: {path}"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, f"Failed to open video file for reading: {path}"

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False, f"Could not decode any frame from video file: {path}"

    return True, ""


def validate_slo(slo_config: Dict[str, Any]) -> List[str]:
    """
    Validate SLO fields for positive values and internal consistency.
    """
    errors: List[str] = []
    slo = slo_config.get("slo", {})
    min_fps = slo.get("min_fps")
    if min_fps is None or not isinstance(min_fps, (int, float)) or min_fps <= 0:
        errors.append(f"SLO min_fps must be positive number. Got: {min_fps}")

    max_p95 = slo.get("max_p95_latency_ms")
    if max_p95 is None or not isinstance(max_p95, (int, float)) or max_p95 <= 0:
        errors.append(f"SLO max_p95_latency_ms must be positive number. Got: {max_p95}")

    min_conf = slo.get("min_detection_confidence")
    if min_conf is None or not isinstance(min_conf, (int, float)) or not (0.0 <= min_conf <= 1.0):
        errors.append(f"SLO min_detection_confidence must be between 0.0 and 1.0. Got: {min_conf}")

    return errors


def validate_single_candidate(
    candidate: Dict[str, Any],
    check_artifacts: bool = True
) -> Tuple[bool, str]:
    """
    Validate parameters and dependencies for a single candidate.

    Returns:
        (is_valid, rejection_reason)
    """
    cand_id = candidate.get("id")
    if not cand_id or not isinstance(cand_id, str):
        return False, "Candidate ID must be a non-empty string."

    # 1. Inference interval check
    interval = candidate.get("inference_interval")
    if interval is None or not isinstance(interval, int) or interval < 1 or interval > 10:
        return False, f"Inference interval must be integer between 1 and 10. Got: {interval}"

    # 2. Resolution check
    res = candidate.get("resolution")
    if res != "native":
        if not (isinstance(res, (list, tuple)) and len(res) == 2):
            return False, f"Resolution must be 'native' or [width, height]. Got: {res}"
        w, h = res
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            return False, f"Resolution dimensions must be positive integers. Got: [{w}, {h}]"

    # 3. ROI mode check
    roi_mode = candidate.get("roi_mode")
    if roi_mode not in ["full_frame", "activity_roi"]:
        return False, f"ROI mode must be 'full_frame' or 'activity_roi'. Got: '{roi_mode}'"

    # 4. PipeGen artifact check
    if check_artifacts:
        cfg_path = candidate.get("pipegen_config_path")
        if not cfg_path or not Path(cfg_path).exists():
            return False, f"Required PipeGen config artifact does not exist: {cfg_path}"

    return True, ""


def validate_candidates(
    candidates: List[Dict[str, Any]],
    slo_config: Dict[str, Any],
    check_artifacts: bool = True,
    check_video: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate candidate list against environment, SLO, and artifact requirements.

    Returns:
        (valid_candidates, invalid_candidates)
    """
    # 1. Global video check
    if check_video:
        video_path = slo_config.get("use_case", {}).get("input_video")
        if not video_path:
            raise ValidationError("Missing 'input_video' in use_case configuration.")
        is_ok, err = is_video_readable(video_path)
        if not is_ok:
            raise ValidationError(f"Input video validation failed: {err}")

    # 2. Global SLO consistency
    slo_errors = validate_slo(slo_config)
    if slo_errors:
        raise ValidationError(f"SLO validation failed: {'; '.join(slo_errors)}")

    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []

    for c in candidates:
        ok, reason = validate_single_candidate(c, check_artifacts=check_artifacts)
        if ok:
            c["status"] = "validated"
            valid.append(c)
        else:
            c["status"] = "invalid"
            c["rejection_reason"] = reason
            invalid.append(c)

    return valid, invalid
