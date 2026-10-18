"""
Metrics collector for AdaPipe.
Captures real-time latency, throughput, quality proxy, and hardware telemetry (GPU/CPU).
Strictly complies with NFR-04 (no fabricated metrics) and NFR-05 (null GPU if unavailable).
"""

from datetime import datetime, timezone
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple
import psutil


def get_gpu_telemetry() -> Tuple[Optional[float], Optional[float]]:
    """
    Query real GPU utilization and memory via nvidia-smi.
    Returns (gpu_utilization_percent, gpu_memory_used_mb) or (None, None) if unavailable.
    """
    if not shutil.which("nvidia-smi"):
        return None, None

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split("\n")[0].split(",")
            if len(parts) >= 2:
                gpu_util = float(parts[0].strip())
                gpu_mem = float(parts[1].strip())
                return gpu_util, gpu_mem
    except Exception:
        pass

    return None, None


def get_cpu_telemetry() -> Tuple[Optional[float], Optional[float]]:
    """
    Query CPU utilization and process memory usage via psutil.
    Returns (cpu_percent, memory_used_mb).
    """
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().used / (1024.0 * 1024.0)
        return round(cpu, 1), round(mem, 1)
    except Exception:
        return None, None


def create_empty_metrics(candidate_id: str) -> Dict[str, Any]:
    """Return a baseline metrics dictionary conforming to Section 5 data model."""
    return {
        "candidate_id": candidate_id,
        "frames_processed": 0,
        "elapsed_seconds": 0.0,
        "fps": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "mean_confidence": 0.0,
        "detection_count": 0,
        "gpu_utilization_percent": None,
        "gpu_memory_used_mb": None,
        "cpu_utilization_percent": None,
        "memory_used_mb": None,
        "started_at": None,
        "ended_at": None,
        "status": "pending",
        "error": None,
    }


def record_run_metrics(
    candidate_id: str,
    pipeline_results: Dict[str, Any],
    started_at: datetime,
    ended_at: datetime,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble the complete benchmark metrics record from pipeline output and system telemetry.
    """
    gpu_util, gpu_mem = get_gpu_telemetry()
    cpu_util, cpu_mem = get_cpu_telemetry()

    status = "failed" if error else "success"

    return {
        "candidate_id": candidate_id,
        "frames_processed": pipeline_results.get("frames_processed", 0),
        "elapsed_seconds": pipeline_results.get("elapsed_seconds", 0.0),
        "fps": pipeline_results.get("fps", 0.0),
        "p50_latency_ms": pipeline_results.get("p50_latency_ms", 0.0),
        "p95_latency_ms": pipeline_results.get("p95_latency_ms", 0.0),
        "mean_confidence": pipeline_results.get("mean_confidence", 0.0),
        "detection_count": pipeline_results.get("detection_count", 0),
        "gpu_utilization_percent": gpu_util,
        "gpu_memory_used_mb": gpu_mem,
        "cpu_utilization_percent": cpu_util,
        "memory_used_mb": cpu_mem,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "status": status,
        "error": error,
    }
