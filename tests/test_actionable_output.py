"""
Tests for AdaPipe Actionable JSON Output & Event Streaming Module.
Verifies event schema, spatial tracking deduplication, JSON/JSONL serialization,
and MQTT graceful fallback.
"""

import json
from pathlib import Path
import pytest

from adapipe.actionable_output import (
    compute_iou,
    compute_centroid,
    determine_spatial_zone,
    SpatialTracker,
    ActionableOutputCollector,
    MQTTPublisher,
)


def test_iou_calculation():
    """Test normalized bounding box IoU calculation."""
    box_a = [0.0, 0.0, 1.0, 1.0]
    box_b = [0.0, 0.0, 1.0, 1.0]
    assert compute_iou(box_a, box_b) == 1.0

    box_c = [0.0, 0.0, 0.5, 0.5]  # area 0.25
    box_d = [0.5, 0.5, 1.0, 1.0]  # area 0.25, no overlap
    assert compute_iou(box_c, box_d) == 0.0

    box_e = [0.0, 0.0, 1.0, 0.5]  # area 0.5
    box_f = [0.0, 0.0, 1.0, 1.0]  # area 1.0, overlap area 0.5, union 1.0
    assert compute_iou(box_e, box_f) == 0.5


def test_spatial_zone_assignment():
    """Test spatial zone assignment logic based on centroid."""
    box_left = [0.2, 0.1, 0.6, 0.2]  # cx = 0.15 -> entryway_left
    assert determine_spatial_zone(box_left) == "entryway_left"

    box_center = [0.2, 0.4, 0.6, 0.6]  # cx = 0.50 -> entryway_center
    assert determine_spatial_zone(box_center) == "entryway_center"

    box_right = [0.2, 0.7, 0.6, 0.9]  # cx = 0.80 -> entryway_right
    assert determine_spatial_zone(box_right) == "entryway_right"

    box_bottom = [0.7, 0.4, 0.9, 0.6]  # cy = 0.80 -> door_threshold
    assert determine_spatial_zone(box_bottom) == "door_threshold"


def test_spatial_tracker_deduplication():
    """
    Test that tracking association preserves identity across consecutive frames
    and does NOT erroneously count the same person repeatedly.
    """
    tracker = SpatialTracker(iou_threshold=0.25)

    # Frame 0: Two distinct people detected
    dets_f0 = [
        {"box": [0.20, 0.10, 0.85, 0.25], "score": 0.88, "class": 0},
        {"box": [0.22, 0.50, 0.82, 0.65], "score": 0.84, "class": 0},
    ]
    res_f0 = tracker.update(dets_f0, frame_idx=0)
    assert len(res_f0) == 2
    assert res_f0[0][2] is True  # is_new_entity
    assert res_f0[1][2] is True  # is_new_entity
    assert len(tracker.unique_entities) == 2

    # Frame 1: Same two people with minor spatial jitter
    dets_f1 = [
        {"box": [0.205, 0.102, 0.852, 0.251], "score": 0.87, "class": 0},
        {"box": [0.218, 0.505, 0.819, 0.652], "score": 0.83, "class": 0},
    ]
    res_f1 = tracker.update(dets_f1, frame_idx=1)
    assert len(res_f1) == 2
    assert res_f1[0][2] is False  # NOT a new entity
    assert res_f1[1][2] is False  # NOT a new entity
    assert len(tracker.unique_entities) == 2  # Count must remain 2!


def test_empty_detections_handled_cleanly():
    """Test that a frame with zero detections produces zero events without errors."""
    tracker = SpatialTracker()
    results = tracker.update([], frame_idx=5)
    assert len(results) == 0

    collector = ActionableOutputCollector()
    frame_events = collector.on_frame({
        "frame_idx": 5,
        "detections": [],
        "frame_shape": [720, 1280],
    })
    assert len(frame_events) == 0
    assert collector.total_detections == 0


def test_actionable_collector_export(tmp_path: Path):
    """Test full JSON and JSONL export from collector."""
    json_path = tmp_path / "actionable_output.json"
    jsonl_path = tmp_path / "actionable_events.jsonl"

    collector = ActionableOutputCollector(source_id="test_video.mp4")

    # Simulate 3 frames with 2 persons
    for f in range(3):
        metrics = {
            "frame_idx": f,
            "latency_ms": 15.2,
            "inference_interval": 1,
            "roi_mode": "full_frame",
            "frame_shape": [720, 1280],
            "detections": [
                {"box": [0.20, 0.10, 0.85, 0.25], "score": 0.88, "class": 0},
                {"box": [0.25, 0.60, 0.86, 0.72], "score": 0.79, "class": 0},
            ],
        }
        collector.on_frame(metrics, fps=30.0)

    collector.export_json(json_path)
    collector.export_jsonl(jsonl_path)

    assert json_path.exists()
    assert jsonl_path.exists()

    # Verify JSON structure
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["source_video"] == "test_video.mp4"
    assert data["summary"]["total_frames_processed"] == 3
    assert data["summary"]["total_detection_events"] == 6
    assert data["summary"]["unique_entities_identified"] == 2
    assert len(data["unique_entities"]) == 2
    assert len(data["events"]) == 6

    # Verify JSONL structure
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    first_ev = json.loads(lines[0])
    assert "event_id" in first_ev
    assert "timestamp" in first_ev
    assert first_ev["object_class"] == "person"
    assert first_ev["tracking_id"] == 1
    assert first_ev["is_new_entity"] is True

    # 3rd event (frame 1) is existing person
    third_ev = json.loads(lines[2])
    assert third_ev["frame_number"] == 1
    assert third_ev["is_new_entity"] is False


def test_mqtt_publisher_offline_fallback():
    """Test that MQTTPublisher handles unavailable broker gracefully without raising exceptions."""
    pub = MQTTPublisher(
        host="127.0.0.1",
        port=9999,  # Unused port
        enabled=True,
    )
    # Should not raise exception even if offline
    published = pub.publish({"test": 123})
    assert published is False
    pub.close()
