# CraftifAI Physical AI Hackathon — Track A: PipeGen
## AdaPipe Product Requirements Document (PRD)

- **Product Name:** AdaPipe
- **Version:** MVP 1.0
- **Owner:** Pavesh Kamalan R
- **Date:** October 18, 2026
- **Track:** CraftifAI Physical AI Hackathon — Track A: PipeGen
- **Hardware Dependency:** NVIDIA GPU (Local GeForce RTX 3050 or Organizer Cloud GPU)

### Core User Stories
- **US-01:** Define a vision requirement and SLO (`configs/slo.yaml`).
- **US-02:** Generate exactly three bounded candidates (`baseline`, `balanced`, `fast`).
- **US-03:** Validate input video readability, numeric ranges, and PipeGen output artifacts prior to benchmarking.
- **US-04:** Benchmark each candidate sequentially on the same video, excluding warmup frames.
- **US-05:** Select the configuration that best satisfies the SLO, with explicit violation scores if unmet.
- **US-06:** Adapt runtime inference intervals dynamically when latency exceeds SLO limits.
- **US-07:** Produce reproducible evidence artifacts (CSV, JSON, JSONL, Markdown).
