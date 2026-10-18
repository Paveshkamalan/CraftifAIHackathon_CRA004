"""
Configuration selector for AdaPipe.
Enforces FR-06 and FR-07 selection algorithm:
- Evaluates SLO compliance per candidate.
- Selects highest confidence candidate among SLO-feasible candidates.
- If none meet SLO, selects least-violating candidate and marks SLO_UNMET.
- Writes selection_report.json and supplies data for Markdown reporting.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def evaluate_candidate_slo(candidate: Dict[str, Any], slo: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate candidate metrics against the SLO constraints and compute normalized violation score.
    """
    metrics = candidate.get("metrics", {})
    fps = metrics.get("fps", 0.0)
    p95 = metrics.get("p95_latency_ms", float("inf"))
    conf = metrics.get("mean_confidence", 0.0)

    min_fps = slo.get("min_fps", 15.0)
    max_p95 = slo.get("max_p95_latency_ms", 250.0)
    min_conf = slo.get("min_detection_confidence", 0.45)

    fps_ok = (fps >= min_fps)
    latency_ok = (p95 <= max_p95)
    quality_ok = (conf >= min_conf)

    slo_met = bool(fps_ok and latency_ok and quality_ok and candidate.get("status") == "success")

    # Normalized violation score: V_c = max(0, (FPS_min - FPS_c)/FPS_min) + max(0, (L_95,c - L_max)/L_max) + max(0, (Q_min - Q_c)/Q_min)
    v_fps = max(0.0, (min_fps - fps) / min_fps) if min_fps > 0 else 0.0
    v_lat = max(0.0, (p95 - max_p95) / max_p95) if max_p95 > 0 else 0.0
    v_qual = max(0.0, (min_conf - conf) / min_conf) if min_conf > 0 else 0.0
    violation_score = round(float(v_fps + v_lat + v_qual), 4)

    eval_result = {
        "candidate_id": candidate["id"],
        "status": candidate.get("status", "unknown"),
        "fps": fps,
        "fps_ok": fps_ok,
        "p95_latency_ms": p95,
        "latency_ok": latency_ok,
        "mean_confidence": conf,
        "quality_ok": quality_ok,
        "slo_status": "SLO_MET" if slo_met else "SLO_UNMET",
        "violation_score": violation_score,
        "inference_interval": candidate.get("inference_interval"),
        "resolution": candidate.get("resolution"),
        "roi_mode": candidate.get("roi_mode"),
    }

    candidate["slo_status"] = eval_result["slo_status"]
    candidate["violation_score"] = violation_score
    return eval_result


def select_best_configuration(
    candidates: List[Dict[str, Any]],
    slo_config: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str, str, List[Dict[str, Any]]]:
    """
    Apply AdaPipe selection algorithm.

    Returns:
        (selected_candidate, overall_slo_status, selection_reason, evaluated_list)
    """
    slo = slo_config.get("slo", {})
    min_fps = slo.get("min_fps", 15.0)
    max_p95 = slo.get("max_p95_latency_ms", 250.0)
    min_conf = slo.get("min_detection_confidence", 0.45)

    evaluated_list: List[Dict[str, Any]] = []
    feasible_candidates: List[Dict[str, Any]] = []

    for c in candidates:
        eval_item = evaluate_candidate_slo(c, slo)
        evaluated_list.append(eval_item)
        if eval_item["slo_status"] == "SLO_MET":
            feasible_candidates.append(c)

    if not candidates:
        return None, "SLO_UNMET", "No candidates evaluated.", []

    if feasible_candidates:
        # Select candidate with highest mean detection confidence
        # Tie-breaker: lowest p95 latency, then highest FPS
        selected = max(
            feasible_candidates,
            key=lambda c: (
                c.get("metrics", {}).get("mean_confidence", 0.0),
                -c.get("metrics", {}).get("p95_latency_ms", float("inf")),
                c.get("metrics", {}).get("fps", 0.0),
            ),
        )
        overall_status = "SLO_MET"
        m = selected.get("metrics", {})
        reason = (
            f"Candidate '{selected['id']}' meets all SLO requirements "
            f"(FPS: {m.get('fps')} >= {min_fps}, p95: {m.get('p95_latency_ms')}ms <= {max_p95}ms, "
            f"confidence: {m.get('mean_confidence')} >= {min_conf}) and achieves highest detection quality."
        )
    else:
        # None meet SLO -> select minimum violation score
        successful_cands = [c for c in candidates if c.get("status") == "success"]
        cands_to_check = successful_cands if successful_cands else candidates

        selected = min(cands_to_check, key=lambda c: c.get("violation_score", float("inf")))
        overall_status = "SLO_UNMET"
        reason = (
            f"No candidate met all SLO constraints. Selected '{selected['id']}' as the least-violating "
            f"configuration with normalized violation score V_c = {selected.get('violation_score', 0.0):.4f}."
        )

    return selected, overall_status, reason, evaluated_list


def create_selection_report(
    candidates: List[Dict[str, Any]],
    slo_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate selection report dictionary matching DOCUMENT 6 schema.
    """
    selected, overall_status, reason, evaluated_list = select_best_configuration(candidates, slo_config)

    selected_dict = None
    if selected:
        selected_dict = {
            "id": selected["id"],
            "inference_interval": selected.get("inference_interval"),
            "resolution": selected.get("resolution"),
            "roi_mode": selected.get("roi_mode"),
            "metrics": selected.get("metrics", {}),
            "slo_status": selected.get("slo_status"),
            "violation_score": selected.get("violation_score", 0.0),
        }

    report = {
        "project": slo_config.get("project", {}).get("name", "AdaPipe"),
        "use_case": slo_config.get("use_case", {}).get("name", "person_detection_entry"),
        "input_video": slo_config.get("use_case", {}).get("input_video", "demo/sample_video.mp4"),
        "slo": slo_config.get("slo", {}),
        "candidates_evaluated": evaluated_list,
        "selected_candidate": selected_dict,
        "selection_reason": reason,
        "overall_slo_status": overall_status,
        "limitations": [
            "The MVP uses one input video and one detection model.",
            "Runtime adaptation is limited to inference interval and ROI mode changes.",
            "GPU metrics are recorded when nvidia-smi is available; otherwise recorded as null.",
            "INT8 precision is reserved for calibrated TensorRT targets; FP16 / FP32 used on local GPU.",
            "ESP32 BLE event source is simulated in this software-only MVP per FirmGen_XX schema.",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return report


def save_selection_report(
    report_data: Dict[str, Any],
    output_path: str | Path = "results/selection_report.json",
) -> Path:
    """Save the selection report to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    return path
