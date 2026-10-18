# AdaPipe Architecture Specification

## Overview

AdaPipe implements a **two-plane control loop** designed to wrap and augment PipeGen-generated video analytics pipelines:

1. **Control Plane:** Responsible for SLO parsing, candidate configuration generation, pre-flight validation, benchmark orchestration, configuration selection, hysteresis-driven adaptation, and evidence generation.
2. **Data Plane:** The PipeGen-generated DeepStream / ONNX Runtime vision pipeline performing video decode, frame scaling, inference execution, object tracking, and bounding-box post-processing.

---

## Architectural Diagram

```mermaid
flowchart TD
    subgraph UserInput["User Inputs & Objectives"]
        YAML["configs/slo.yaml<br/>(Min FPS, Max p95 Latency, Min Conf)"]
        CandidatesYAML["configs/candidates.yaml<br/>(3 MVP Candidates)"]
        Video["demo/sample_video.mp4<br/>demo/high_load_video.mp4"]
    end

    subgraph ControlPlane["AdaPipe Control Plane (Python)"]
        ConfigLoader["config_loader.py<br/>(Schema & Constraint Validator)"]
        CandGen["candidate_generator.py<br/>(Bounded Search Space)"]
        Validator["validator.py<br/>(Pre-flight Feasibility Check)"]
        BenchRunner["benchmark_runner.py<br/>(Sequential Isolated Orchestrator)"]
        MetricsCol["metrics_collector.py<br/>(GPU, CPU, FPS, Latency p50/p95)"]
        Selector["selector.py<br/>(SLO Feasibility & Violation Scoring)"]
        PolicyEngine["runtime_policy.py<br/>(Hysteresis & Quality Guardrails)"]
        Reporter["report.py<br/>(Markdown, JSON & CSV Artifacts)"]
    end

    subgraph DataPlane["PipeGen Data Plane (DeepStream / ONNX)"]
        PipeConfig["pipegen_output/generated_pipeline/config.yaml"]
        Model["pipegen_output/generated_pipeline/model.onnx"]
        PipelineRunner["pipegen_output/generated_pipeline/pipeline.py<br/>(Decode ➔ Scale ➔ Infer ➔ Track)"]
    end

    subgraph EvidenceArtifacts["Evidence Artifacts"]
        CSV["results/benchmark_results.csv"]
        JSON["results/selection_report.json"]
        JSONL["results/adaptation_log.jsonl"]
        MD["results/selection_report.md"]
        UI["adapipe/dashboard.py (Streamlit UI)"]
    end

    YAML --> ConfigLoader
    CandidatesYAML --> CandGen
    ConfigLoader --> Validator
    CandGen --> Validator
    Video --> Validator

    Validator -->|Valid Candidates| BenchRunner
    PipeConfig --> PipelineRunner
    Model --> PipelineRunner

    BenchRunner -->|Executes Sequentially| PipelineRunner
    PipelineRunner -->|Raw Per-frame Timers| MetricsCol
    MetricsCol --> CSV
    MetricsCol --> Selector

    Selector -->|Selected Configuration| PolicyEngine
    Selector --> JSON
    Selector --> Reporter

    PolicyEngine -->|Dynamic Interval Update| PipelineRunner
    PolicyEngine --> JSONL
    
    CSV --> Reporter
    JSON --> Reporter
    JSONL --> Reporter
    Reporter --> MD
    Reporter --> UI
```

---

## Control Plane vs. Data Plane Separation

| Plane | Component | Language / Framework | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Control Plane** | `config_loader` | Python 3.10+ / PyYAML | Loads SLO constraints (`min_fps`, `max_p95_latency_ms`, `min_detection_confidence`). |
| **Control Plane** | `candidate_generator` | Python 3.10+ | Instantiates the bounded candidate matrix (baseline, balanced, fast). |
| **Control Plane** | `validator` | Python 3.10+ / OpenCV | Verifies video readability, numeric validity, and PipeGen artifact presence. |
| **Control Plane** | `benchmark_runner` | Python 3.10+ | Runs each candidate sequentially, isolates runs, ignores warmup period. |
| **Control Plane** | `metrics_collector` | Python 3.10+ / psutil / nvidia-smi | Captures p50/p95 latency, FPS, GPU util %, GPU memory MB, CPU %. |
| **Control Plane** | `selector` | Python 3.10+ | Evaluates SLO feasibility and calculates normalized violation score $V_c$. |
| **Control Plane** | `runtime_policy` | Python 3.10+ | Evaluates 5-window rolling telemetry; applies degradation and recovery hysteresis. |
| **Control Plane** | `report` | Python 3.10+ / pandas | Generates reproducible CSV, JSON, JSONL, and Markdown evidence. |
| **Data Plane** | `pipeline.py` | OpenCV / ONNX Runtime / CUDA / DirectML | Decodes input stream, scales resolution, executes ONNX neural network, filters NMS. |

---

## Hysteresis & Guardrail State Machine

```mermaid
stateDiagram-v2
    [*] --> SteadyState: Initial Configuration Selected

    state SteadyState {
        [*] --> Monitoring
        Monitoring --> LatencyExceeded: p95 > SLO Max
        LatencyExceeded --> Monitoring: Resolved within < 3 windows
        LatencyExceeded --> DegradeTrigger: 3 Consecutive Violations
        
        Monitoring --> LowLatency: p95 < 0.7 * SLO Max
        LowLatency --> Monitoring: Latency increases within < 5 windows
        LowLatency --> UpgradeTrigger: 5 Consecutive Low-load Windows
    }

    DegradeTrigger --> DegradedState: Degrade 1 Level (1 ➔ 2 ➔ 3)
    
    state DegradedState {
        [*] --> CheckQuality
        CheckQuality --> RevertTrigger: Confidence Drop > 20%
        CheckQuality --> StableDegraded: Quality Preserved
    }

    RevertTrigger --> SteadyState: Quality Guardrail Reverts Configuration
    UpgradeTrigger --> SteadyState: Recover Configuration (3 ➔ 2 ➔ 1)
```
