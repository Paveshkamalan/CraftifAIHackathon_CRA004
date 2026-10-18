"""
Software-only simulated BLE event source for AdaPipe.
Emits events matching the ESP32 FirmGen_XX advertisement schema.
Note: Per NFR-09 and hackathon discipline, this module is software-simulated
and explicitly does NOT claim physical BLE hardware integration.
"""

from datetime import datetime, timezone
import json
import random
import time
from typing import Any, Callable, Dict, Generator, Optional


class SimulatedBLEEventSource:
    """
    Simulates Bluetooth Low Energy (BLE) advertisements from an ESP32 sensor node.
    Can trigger perception pipeline activation or stream prioritization.
    """

    def __init__(
        self,
        device_id: str = "ESP32_FirmGen_01",
        zone: str = "entrance_door_1",
    ):
        self.device_id = device_id
        self.zone = zone
        self.packet_seq = 0

    def generate_event(
        self,
        event_type: str = "motion_detected",
        rssi: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a single ESP32 BLE advertisement packet."""
        self.packet_seq += 1
        return {
            "device_id": self.device_id,
            "seq": self.packet_seq,
            "event_type": event_type,
            "zone": self.zone,
            "rssi_dbm": rssi if rssi is not None else random.randint(-75, -50),
            "battery_mv": 3250 + random.randint(-20, 20),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware_type": "ESP32-S3 (Simulated)",
            "simulated": True,
        }

    def event_stream(
        self,
        count: int = 5,
        interval_seconds: float = 1.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream simulated BLE events over time."""
        for _ in range(count):
            yield self.generate_event()
            time.sleep(interval_seconds)

    def listen(
        self,
        callback: Callable[[Dict[str, Any]], None],
        max_events: int = 5,
        interval_seconds: float = 0.5,
    ) -> None:
        """Publish events to a listener callback."""
        for evt in self.event_stream(count=max_events, interval_seconds=interval_seconds):
            callback(evt)
