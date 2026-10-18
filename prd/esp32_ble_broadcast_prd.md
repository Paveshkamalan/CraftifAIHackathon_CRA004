# ESP32 BLE Broadcast Specification (FirmGen_XX Schema)

## Purpose
Specifies the advertisement packet schema emitted by simulated or physical ESP32 BLE peripheral nodes to trigger perception pipeline prioritization in Physical AI edge installations.

## Schema Specification
```json
{
  "device_id": "ESP32_FirmGen_01",
  "seq": 104,
  "event_type": "motion_detected",
  "zone": "entrance_door_1",
  "rssi_dbm": -62,
  "battery_mv": 3260,
  "timestamp": "2026-10-18T10:45:00.000Z",
  "hardware_type": "ESP32-S3 (Simulated)",
  "simulated": true
}
```

## Hardware Discipline
Per Hackathon rules, in this software-only MVP, ESP32 BLE packet emission is simulated via `adapipe/ble_event_source.py` and clearly documented as software simulation without claiming physical hardware integration.
