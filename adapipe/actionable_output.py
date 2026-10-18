"""
Actionable JSON Output and Event Streaming for AdaPipe.
Track A: PipeGen - CraftifAI Physical AI Hackathon.

Captures real-time detection metadata from the PipeGen vision pipeline,
performs spatial track association to avoid duplicate counting of consecutive frames,
emits structured JSON events (actionable_output.json and actionable_events.jsonl),
and optionally publishes events to an MQTT broker.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger("adapipe.actionable_output")


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) for two normalized boxes [ymin, xmin, ymax, xmax].
    """
    y1 = max(box_a[0], box_b[0])
    x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])

    inter_h = max(0.0, y2 - y1)
    inter_w = max(0.0, x2 - x1)
    inter_area = inter_h * inter_w

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_centroid(box: List[float]) -> Tuple[float, float]:
    """Compute normalized centroid (cy, cx) for box [ymin, xmin, ymax, xmax]."""
    cy = (box[0] + box[2]) / 2.0
    cx = (box[1] + box[3]) / 2.0
    return cy, cx


def determine_spatial_zone(box: List[float]) -> str:
    """Determine spatial zone of the detected person based on normalized centroid."""
    cy, cx = compute_centroid(box)
    if cy >= 0.70:
        return "door_threshold"
    elif cx < 0.35:
        return "entryway_left"
    elif cx > 0.65:
        return "entryway_right"
    else:
        return "entryway_center"


class SpatialTracker:
    """
    Lightweight spatial track association module.
    Associates bounding boxes across consecutive frames using IoU and centroid distance.
    Prevents duplicate counting of the same person across continuous observations.
    """

    def __init__(self, iou_threshold: float = 0.25, max_missed_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.unique_entities: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
    ) -> List[Tuple[Dict[str, Any], int, bool]]:
        """
        Update tracker with detections from current frame.
        Returns: list of (detection, tracking_id, is_new_entity).
        """
        matched_results = []
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        # Attempt to match current detections with active tracks using IoU
        for d_idx, det in enumerate(detections):
            best_track_id = None
            best_iou = -1.0

            for t_id, track in self.active_tracks.items():
                if t_id in matched_tracks:
                    continue
                iou = compute_iou(det["box"], track["last_box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = t_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                # Matched with existing track
                matched_tracks.add(best_track_id)
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)

                track = self.active_tracks[best_track_id]
                track["last_box"] = det["box"]
                track["last_frame"] = frame_idx
                track["total_hits"] += 1
                track["scores"].append(det["score"])
                track["missed_frames"] = 0

                # Not a new entity
                matched_results.append((det, best_track_id, False))

        # Any remaining unmatched detections are new entities
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[new_id] = {
                "track_id": new_id,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "last_box": det["box"],
                "total_hits": 1,
                "scores": [det["score"]],
                "missed_frames": 0,
                "class": det.get("class", 0),
            }
            self.unique_entities[new_id] = self.active_tracks[new_id]
            matched_results.append((det, new_id, True))

        # Prune tracks that have not been seen for max_missed_frames
        dead_tracks = []
        for t_id, track in self.active_tracks.items():
            if t_id not in matched_tracks and t_id not in [r[1] for r in matched_results]:
                track["missed_frames"] += 1
                if track["missed_frames"] > self.max_missed_frames:
                    dead_tracks.append(t_id)

        for t_id in dead_tracks:
            del self.active_tracks[t_id]

        return matched_results


class MQTTPublisher:
    """
    Optional MQTT publisher for real-time edge telemetry.
    If paho-mqtt or the broker is unavailable, fails gracefully without disrupting core pipeline.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        topic: str = "adapipe/events/person_detection",
        client_id: str = "adapipe_edge_publisher",
        enabled: bool = False,
    ):
        self.host = host
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.enabled = enabled
        self.connected = False
        self.client: Any = None

        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt  # type: ignore

            self.client = mqtt.Client(client_id=self.client_id)
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            self.connected = True
            logger.info("Connected to MQTT broker at %s:%d (topic=%s)", self.host, self.port, self.topic)
        except ImportError:
            logger.warning("[MQTT] 'paho-mqtt' is not installed; MQTT live publishing disabled.")
            self.connected = False
        except Exception as exc:
            logger.warning("[MQTT] Broker connection failed (%s:%d): %s; publishing disabled.", self.host, self.port, exc)
            self.connected = False

    def publish(self, event: Dict[str, Any]) -> bool:
        """Publish a single structured event payload to the configured topic."""
        if not self.enabled or not self.connected or self.client is None:
            return False
        try:
            payload = json.dumps(event)
            self.client.publish(self.topic, payload, qos=0)
            return True
        except Exception as exc:
            logger.warning("[MQTT] Failed to publish event: %s", exc)
            return False

    def close(self) -> None:
        """Disconnect and stop background network loop."""
        if self.connected and self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            self.connected = False


class ActionableOutputCollector:
    """
    Collects detection frames from PipeGen pipeline, assigns tracking IDs,
    constructs structured actionable events, and exports JSON / JSONL artifacts.
    """

    def __init__(
        self,
        source_id: str = "demo/sample_video.mp4",
        model_name: str = "person_detector_entry",
        mqtt_config: Optional[Dict[str, Any]] = None,
    ):
        self.source_id = source_id
        self.model_name = model_name
        self.tracker = SpatialTracker(iou_threshold=0.25, max_missed_frames=10)

        # Optional MQTT
        mqtt_cfg = mqtt_config or {}
        self.mqtt_publisher = MQTTPublisher(
            host=mqtt_cfg.get("host", "localhost"),
            port=mqtt_cfg.get("port", 1883),
            topic=mqtt_cfg.get("topic", "adapipe/events/person_detection"),
            client_id=mqtt_cfg.get("client_id", "adapipe_edge_publisher"),
            enabled=mqtt_cfg.get("enabled", False),
        )

        self.events: List[Dict[str, Any]] = []
        self.total_frames = 0
        self.total_detections = 0
        self.zone_counts: Dict[str, int] = {}
        self.start_wall_time = datetime.now(timezone.utc)

    def on_frame(self, metrics: Dict[str, Any], fps: float = 30.0) -> List[Dict[str, Any]]:
        """
        Process a single frame output dictionary from PipeGen pipeline.
        Extracts detections, performs tracking, creates events, and logs.
        """
        frame_idx = metrics.get("frame_idx", self.total_frames)
        self.total_frames += 1

        detections = metrics.get("detections", [])
        frame_shape = metrics.get("frame_shape", [720, 1280])
        h, w = frame_shape[0], frame_shape[1]

        # Update tracker
        tracked = self.tracker.update(detections, frame_idx)

        frame_events = []
        now_iso = datetime.now(timezone.utc).isoformat()
        video_offset_sec = round(float(frame_idx) / fps, 3) if fps > 0 else 0.0

        for det_idx, (det, track_id, is_new) in enumerate(tracked):
            self.total_detections += 1
            box = det["box"]
            score = float(det["score"])
            zone = determine_spatial_zone(box)
            self.zone_counts[zone] = self.zone_counts.get(zone, 0) + 1

            # Pixel coords [ymin, xmin, ymax, xmax] -> [x1, y1, x2, y2]
            y1_px = int(round(box[0] * h))
            x1_px = int(round(box[1] * w))
            y2_px = int(round(box[2] * h))
            x2_px = int(round(box[3] * w))

            event_id = f"evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_f{frame_idx:04d}_t{track_id:02d}_{det_idx:02d}"

            event = {
                "event_id": event_id,
                "timestamp": now_iso,
                "frame_number": frame_idx,
                "video_time_offset_sec": video_offset_sec,
                "source": self.source_id,
                "event_type": "person_detected",
                "object_class": "person",
                "confidence": round(score, 4),
                "bounding_box": [round(float(c), 4) for c in box],
                "bounding_box_pixels": [x1_px, y1_px, x2_px, y2_px],
                "tracking_id": track_id,
                "is_new_entity": is_new,
                "spatial_zone": zone,
                "pipeline_metadata": {
                    "inference_interval": metrics.get("inference_interval", 1),
                    "roi_mode": metrics.get("roi_mode", "full_frame"),
                    "latency_ms": round(metrics.get("latency_ms", 0.0), 2),
                    "model_name": self.model_name,
                },
            }

            self.events.append(event)
            frame_events.append(event)

            # Optional MQTT publication
            if self.mqtt_publisher.enabled:
                self.mqtt_publisher.publish(event)

        return frame_events

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level metadata and tracking summary."""
        now_iso = datetime.now(timezone.utc).isoformat()
        unique_entity_list = []
        for t_id, data in self.tracker.unique_entities.items():
            scores = data.get("scores", [0.0])
            mean_score = sum(scores) / len(scores) if scores else 0.0
            primary_zone = determine_spatial_zone(data["last_box"])
            unique_entity_list.append({
                "tracking_id": t_id,
                "object_class": "person",
                "first_seen_frame": data["first_frame"],
                "last_seen_frame": data["last_frame"],
                "total_observations": data["total_hits"],
                "mean_confidence": round(mean_score, 4),
                "primary_zone": primary_zone,
                "bounding_box_sample": [round(float(c), 4) for c in data["last_box"]],
            })

        mean_conf = (
            sum(e["confidence"] for e in self.events) / len(self.events)
            if self.events
            else 0.0
        )

        return {
            "metadata": {
                "project": "AdaPipe",
                "version": "1.0.0",
                "track": "CraftifAI Physical AI Hackathon — Track A (PipeGen)",
                "use_case": "door_person_count",
                "source_video": self.source_id,
                "generated_at": now_iso,
                "model_name": self.model_name,
                "tracking_method": "spatial_iou_centroid_association",
                "directional_tracking_status": "uncalibrated_line_crossing_omitted",
                "mqtt_publishing": {
                    "enabled": self.mqtt_publisher.enabled,
                    "connected": self.mqtt_publisher.connected,
                    "broker": f"{self.mqtt_publisher.host}:{self.mqtt_publisher.port}",
                    "topic": self.mqtt_publisher.topic,
                },
            },
            "summary": {
                "total_frames_processed": self.total_frames,
                "total_detection_events": len(self.events),
                "unique_entities_identified": len(self.tracker.unique_entities),
                "class_distribution": {"person": len(self.events)},
                "mean_detection_confidence": round(mean_conf, 4),
                "spatial_zones_detected": self.zone_counts,
            },
            "unique_entities": unique_entity_list,
            "sample_events": self.events[:10],
        }

    def export_json(self, output_path: str | Path = "results/actionable_output.json") -> Path:
        """Export structured summary and event manifest to JSON."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        summary_data = self.get_summary()
        summary_data["events"] = self.events

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        logger.info("Saved actionable JSON summary to %s (%d events)", out_file, len(self.events))
        return out_file

    def export_jsonl(self, output_path: str | Path = "results/actionable_events.jsonl") -> Path:
        """Export line-delimited JSON (NDJSON/JSONL) with one event per line."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev) + "\n")

        logger.info("Saved actionable JSONL stream to %s (%d lines)", out_file, len(self.events))
        return out_file

    def close(self) -> None:
        """Clean up background connections."""
        self.mqtt_publisher.close()


def generate_actionable_events_from_video(
    video_path: str = "demo/sample_video.mp4",
    config_path: str = "pipegen_output/generated_pipeline/config.yaml",
    model_path: str = "pipegen_output/generated_pipeline/model.onnx",
    json_output: str = "results/actionable_output.json",
    jsonl_output: str = "results/actionable_events.jsonl",
    duration_seconds: float = 10.0,
    warmup_seconds: float = 1.0,
    inference_interval: int = 1,
    resolution: str = "native",
    mqtt_enabled: bool = False,
    mqtt_host: str = "localhost",
    mqtt_port: int = 1883,
    mqtt_topic: str = "adapipe/events/person_detection",
) -> Dict[str, Any]:
    """
    End-to-end execution function: processes input video through the PipeGen pipeline,
    extracts genuine detection metadata, and generates actionable JSON and JSONL artifacts.
    """
    from pipegen_output.generated_pipeline.pipeline import PipeGenPipeline

    collector = ActionableOutputCollector(
        source_id=video_path,
        model_name="person_detector_entry",
        mqtt_config={
            "enabled": mqtt_enabled,
            "host": mqtt_host,
            "port": mqtt_port,
            "topic": mqtt_topic,
        },
    )

    pipeline = PipeGenPipeline(
        config_path=config_path,
        model_path=model_path,
        inference_interval=inference_interval,
        resolution=resolution,
    )

    logger.info("Executing PipeGen pipeline on %s (duration=%ss, warmup=%ss)", video_path, duration_seconds, warmup_seconds)

    # Run pipeline with collector callback
    pipeline.run(
        video_path=video_path,
        duration_seconds=duration_seconds,
        warmup_seconds=warmup_seconds,
        callback=lambda metrics: collector.on_frame(metrics, fps=30.0),
    )

    collector.export_json(json_output)
    collector.export_jsonl(jsonl_output)
    summary = collector.get_summary()
    collector.close()

    return summary
