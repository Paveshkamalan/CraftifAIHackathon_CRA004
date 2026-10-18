"""
Runtime adaptation policy for AdaPipe.
Implements FR-08, FR-09:
- Rolling window monitoring.
- Hysteresis-based degradation (3 consecutive violations) and recovery (5 consecutive low-load windows).
- Quality preservation guardrail (revert if confidence drops > 20% after degradation).
- JSONL structured event logging to results/adaptation_log.jsonl.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class RuntimePolicy:
    """
    Manages runtime adaptation rules, rolling metric windows,
    and hysteresis counters for PipeGen pipelines.
    """

    def __init__(
        self,
        initial_interval: int = 1,
        max_p95_latency_ms: float = 250.0,
        degrade_after_violations: int = 3,
        recover_after_success_windows: int = 5,
        recovery_latency_ratio: float = 0.7,
        quality_drop_revert_threshold: float = 0.20,
        window_size: int = 5,
        log_file: Optional[str | Path] = "results/adaptation_log.jsonl",
    ):
        self.initial_interval = max(1, initial_interval)
        self.current_interval = self.initial_interval
        self.previous_interval = self.initial_interval
        self.max_p95_latency_ms = float(max_p95_latency_ms)

        self.degrade_threshold = degrade_after_violations
        self.recover_threshold = recover_after_success_windows
        self.recovery_ratio = recovery_latency_ratio
        self.quality_drop_threshold = quality_drop_revert_threshold
        self.window_size = window_size

        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.consecutive_violations = 0
        self.consecutive_recoveries = 0
        self.pre_degradation_confidence: Optional[float] = None
        self.just_degraded = False

        self.window_latencies: List[float] = []
        self.window_confidences: List[float] = []
        self.event_history: List[Dict[str, Any]] = []

    def _log_event(self, event_data: Dict[str, Any]) -> None:
        self.event_history.append(event_data)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data) + "\n")

    def process_window(
        self,
        window_p95: float,
        window_mean_conf: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a completed metric window against degradation, recovery, and quality guardrail rules.
        """
        now = datetime.now(timezone.utc).isoformat()

        # 1. Quality Guardrail Check: if confidence collapsed > 20% right after degradation
        if self.just_degraded and self.pre_degradation_confidence is not None and self.pre_degradation_confidence > 0:
            conf_drop = (self.pre_degradation_confidence - window_mean_conf) / self.pre_degradation_confidence
            if conf_drop > self.quality_drop_threshold:
                reverted_from = self.current_interval
                self.current_interval = self.previous_interval
                self.just_degraded = False
                self.consecutive_violations = 0
                self.consecutive_recoveries = 0

                event = {
                    "timestamp": now,
                    "event": "quality_guardrail_triggered",
                    "previous_interval": reverted_from,
                    "reverted_interval": self.current_interval,
                    "baseline_confidence": round(self.pre_degradation_confidence, 4),
                    "current_confidence": round(window_mean_conf, 4),
                    "confidence_drop_ratio": round(conf_drop, 4),
                    "threshold_ratio": self.quality_drop_threshold,
                    "reason": f"Mean confidence dropped by {conf_drop*100:.1f}% (> {self.quality_drop_threshold*100}% threshold) after degradation; reverted to interval {self.current_interval}.",
                }
                self._log_event(event)
                return event

        self.just_degraded = False

        # 2. Degrade Check: 3 consecutive windows exceeding max_p95_latency_ms
        if window_p95 > self.max_p95_latency_ms:
            self.consecutive_violations += 1
            self.consecutive_recoveries = 0

            if self.consecutive_violations >= self.degrade_threshold:
                if self.current_interval < 3:  # 1 -> 2 -> 3
                    prev = self.current_interval
                    self.current_interval += 1
                    self.previous_interval = prev
                    self.pre_degradation_confidence = window_mean_conf
                    self.just_degraded = True
                    self.consecutive_violations = 0

                    event = {
                        "timestamp": now,
                        "event": "degrade_configuration",
                        "previous_interval": prev,
                        "new_interval": self.current_interval,
                        "consecutive_violations": self.degrade_threshold,
                        "current_p95_latency_ms": round(window_p95, 2),
                        "slo_max_p95_latency_ms": self.max_p95_latency_ms,
                        "reason": f"p95 latency ({window_p95:.1f}ms) exceeded SLO target ({self.max_p95_latency_ms:.1f}ms) for {self.degrade_threshold} consecutive windows.",
                    }
                    self._log_event(event)
                    return event
        else:
            self.consecutive_violations = 0

        # 3. Recovery Check: 5 consecutive windows with p95 < 0.7 * max_p95_latency_ms
        recovery_limit = self.recovery_ratio * self.max_p95_latency_ms
        if window_p95 < recovery_limit:
            self.consecutive_recoveries += 1
            if self.consecutive_recoveries >= self.recover_threshold:
                # MVP rule: Shall not upgrade beyond initial selected configuration
                if self.current_interval > self.initial_interval:
                    prev = self.current_interval
                    self.current_interval -= 1
                    self.previous_interval = prev
                    self.consecutive_recoveries = 0

                    event = {
                        "timestamp": now,
                        "event": "recover_configuration",
                        "previous_interval": prev,
                        "new_interval": self.current_interval,
                        "consecutive_recovery_windows": self.recover_threshold,
                        "current_p95_latency_ms": round(window_p95, 2),
                        "recovery_threshold_ms": round(recovery_limit, 2),
                        "reason": f"p95 latency ({window_p95:.1f}ms) remained below {self.recovery_ratio*100:.0f}% of SLO target ({recovery_limit:.1f}ms) for {self.recover_threshold} consecutive windows.",
                    }
                    self._log_event(event)
                    return event
        else:
            self.consecutive_recoveries = 0

        return None

    def record_sample(
        self,
        latency_ms: float,
        confidence: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Record a single frame sample. When rolling window accumulates window_size samples,
        computes statistics and evaluates policy.
        """
        self.window_latencies.append(latency_ms)
        self.window_confidences.append(confidence)

        if len(self.window_latencies) >= self.window_size:
            p95 = float(np.percentile(self.window_latencies, 95))
            mean_conf = float(np.mean(self.window_confidences))

            # Slide window
            self.window_latencies = []
            self.window_confidences = []

            return self.process_window(p95, mean_conf)

        return None


def run_adaptive_pipeline(
    candidate: Dict[str, Any],
    video_path: str,
    slo_config: Dict[str, Any],
    duration_seconds: float = 30.0,
    warmup_seconds: float = 3.0,
    synthetic_high_load_stress: bool = False,
) -> Dict[str, Any]:
    """
    Run the selected candidate pipeline under live runtime monitoring and adaptation.
    """
    from pipegen_output.generated_pipeline.pipeline import PipeGenPipeline

    slo = slo_config.get("slo", {})
    rp_cfg = slo_config.get("runtime_policy", {})
    log_path = slo_config.get("output", {}).get("adaptation_log", "results/adaptation_log.jsonl")

    init_interval = candidate.get("inference_interval", 1)

    policy = RuntimePolicy(
        initial_interval=init_interval,
        max_p95_latency_ms=slo.get("max_p95_latency_ms", 250.0),
        degrade_after_violations=rp_cfg.get("degrade_after_violations", 3),
        recover_after_success_windows=rp_cfg.get("recover_after_success_windows", 5),
        recovery_latency_ratio=rp_cfg.get("recovery_latency_ratio", 0.70),
        quality_drop_revert_threshold=rp_cfg.get("quality_drop_revert_threshold", 0.20),
        window_size=5,
        log_file=log_path,
    )

    pipeline = PipeGenPipeline(
        config_path=candidate.get("pipegen_config_path", "pipegen_output/generated_pipeline/config.yaml"),
        inference_interval=init_interval,
        resolution=candidate.get("resolution", "native"),
        roi_mode=candidate.get("roi_mode", "full_frame"),
    )

    adaptation_events = []

    def frame_callback(metrics: Dict[str, Any]):
        nonlocal adaptation_events
        lat = metrics["latency_ms"]
        conf = metrics["mean_confidence"]

        # If stress mode requested (e.g. high load video simulation trigger)
        if synthetic_high_load_stress:
            # Emulate transient system contention / high crowd density latency spike
            # Frame latency spikes above max_p95 (250ms) when interval is 1, then reduces as interval increases
            interval_factor = {1: 320.0, 2: 190.0, 3: 85.0}.get(policy.current_interval, 300.0)
            lat = max(lat, interval_factor)

        event = policy.record_sample(lat, conf)
        if event:
            adaptation_events.append(event)
            # Live adaptation: propagate degraded or upgraded interval to executing pipeline
            pipeline.set_inference_interval(policy.current_interval)
            print(f"    [ADAPTATION EVENT] {event['event']} -> interval={policy.current_interval}: {event['reason']}")

    print(f"--> Running adaptive pipeline for '{candidate['id']}' on {video_path}...")
    pipeline_results = pipeline.run(
        video_path=video_path,
        duration_seconds=duration_seconds,
        warmup_seconds=warmup_seconds,
        callback=frame_callback,
    )

    return {
        "candidate_id": candidate["id"],
        "pipeline_results": pipeline_results,
        "final_interval": policy.current_interval,
        "adaptation_events": adaptation_events,
        "log_file": str(log_path),
    }
