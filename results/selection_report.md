# AdaPipe Configuration Selection & Benchmark Evidence Report

**Generated at:** `2026-10-18T10:43:37.296743+00:00`  
**Track:** CraftifAI Physical AI Hackathon — Track A: PipeGen  
**Use Case:** `person_detection_entry`  
**Input Video:** `demo/sample_video.mp4`  
**Overall SLO Status:** **`SLO_MET`**

## 1. Service Level Objectives (SLO)

| Metric | Target Constraint | Description |
| :--- | :--- | :--- |
| **Minimum FPS** | `≥ 15 FPS` | Throughput target for real-time video stream ingestion |
| **Maximum p95 Latency** | `≤ 250 ms` | Upper bound for tail frame processing latency |
| **Minimum Detection Confidence** | `≥ 0.45` | Minimum acceptable mean detection confidence score |

## 2. Benchmark Evaluation Summary

| Candidate | FPS | p50 Latency (ms) | p95 Latency (ms) | Mean Conf | Detections | GPU Util (%) | CPU Util (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline** | 15.19 | 34.88 | 123.07 | 0.7030 | 3000 | 31.0% | 52.7% | `success` |
| balanced | 14.33 | 46.97 | 132.55 | 0.6960 | 3000 | 34.0% | 52.4% | `success` |
| fast | 10.01 | 60.73 | 135.15 | 0.6936 | 3000 | 47.0% | 46.4% | `success` |


## 3. Configuration Selection Decision

### Selected Configuration: **`baseline`**
- **Inference Interval:** `1`
- **Resolution:** `native`
- **ROI Mode:** `full_frame`
- **SLO Compliance:** `SLO_MET`
- **Normalized Violation Score (V_c):** `0.0000`

> **Technical Selection Rationale:**  
> Candidate 'baseline' meets all SLO requirements (FPS: 15.19 >= 15, p95: 123.07ms <= 250ms, confidence: 0.703 >= 0.45) and achieves highest detection quality.

## 4. Runtime Adaptation & Guardrail Log

| Timestamp | Event Type | Prev Interval | New Interval | Details |
| :--- | :--- | :---: | :---: | :--- |
| `2026-10-18T10:38:03.004168+00:00` | `degrade_configuration` | 1 | 2 | p95 latency (320.0ms) exceeded SLO target (250.0ms) for 3 consecutive windows. |
| `2026-10-18T10:43:48.548008+00:00` | `degrade_configuration` | 1 | 2 | p95 latency (320.0ms) exceeded SLO target (250.0ms) for 3 consecutive windows. |


## 5. Architectural Boundaries & Limitations

- **Single Model Scope:** The MVP evaluates one ONNX person detection model across resolution and interval variants.
- **Hysteresis Thresholds:** Degradation triggers after 3 consecutive latency violations; recovery requires 5 consecutive low-load windows.
- **Quality Preservation Guardrail:** If mean confidence drops > 20% following a downgrade event, the system reverts immediately to protect perception accuracy.
- **Telemetry Discipline:** Per NFR-04 and NFR-05, all metrics reflect actual hardware counters; unavailable GPU metrics are recorded as null without fabrication.

## 6. Reproduction Commands

```powershell
# 1. Run full benchmark suite
python -m adapipe.main --mode benchmark

# 2. Run selection engine
python -m adapipe.main --mode select

# 3. Run adaptive execution on high-load video
python -m adapipe.main --mode adaptive

# 4. Generate evidence report
python -m adapipe.main --mode report
```
