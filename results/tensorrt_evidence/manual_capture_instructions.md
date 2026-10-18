# Manual Evidence Capture Instructions for Hackathon Judges & Evaluators

This document details the exact sequence of commands and inspection procedures to reproduce TensorRT engine compilation and performance benchmarking on a supported NVIDIA platform (e.g., NVIDIA GeForce RTX 3050 Laptop GPU, NVIDIA RTX 4090, or NVIDIA Jetson AGX Orin).

---

## 1. Environment Verification
Execute the following commands to confirm that NVIDIA driver, CUDA, and TensorRT are loaded:

```bash
# 1. Inspect NVIDIA GPU and Driver
nvidia-smi

# 2. Check CUDA compiler version
nvcc --version

# 3. Check TensorRT execution utility
trtexec --version
```

**What to capture:**
- Take a screenshot of the terminal displaying `nvidia-smi` GPU status table showing the GPU model, VRAM, and driver version.
- Save as `results/tensorrt_evidence/nvidia_smi.png`.

---

## 2. TensorRT Engine Compilation
Run the automated compilation script or execute `trtexec` directly:

```bash
# Execute compilation script
chmod +x results/tensorrt_evidence/compile_tensorrt.sh
./results/tensorrt_evidence/compile_tensorrt.sh
```

Or run manual compilation:
```bash
# FP16 Precision Compilation
trtexec --onnx=pipegen_output/generated_pipeline/model.onnx \
        --saveEngine=results/tensorrt_evidence/model_fp16.engine \
        --fp16 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:1x3x640x640 \
        --buildOnly \
        --verbose > results/tensorrt_evidence/trtexec_compile_fp16.log 2>&1
```

**What to capture:**
- Screenshot of the terminal output showing `[MemUsageChange] TensorRT-managed allocation in engine building`, `Engine generation completed`, and the engine file size on disk (`ls -lh results/tensorrt_evidence/model_fp16.engine`).
- Save as `results/tensorrt_evidence/trtexec_compilation.png`.

---

## 3. Raw Engine Performance Profiling
Run standalone profiling using `trtexec` with a 5-second warmup:

```bash
trtexec --loadEngine=results/tensorrt_evidence/model_fp16.engine \
        --warmUp=5000 \
        --duration=45 \
        --iterations=1000 \
        --percentile=50,95,99 \
        --dumpProfile \
        --exportTimes=results/tensorrt_evidence/benchmark_times_fp16.json \
        > results/tensorrt_evidence/benchmark_fp16.log 2>&1
```

**What to capture:**
- Screenshot of the summary block from `trtexec`:
  - `Throughput: ... qps`
  - `Latency: min = ... ms, max = ... ms, mean = ... ms, median = ... ms, percentile(95%) = ... ms, percentile(99%) = ... ms`
  - `GPU Compute Time: min = ... ms, max = ... ms, mean = ... ms`
- Save as `results/tensorrt_evidence/trtexec_benchmark.png`.

---

## 4. DeepStream Pipeline Integration
To test the engine integrated within the PipeGen DeepStream pipeline:

```bash
# Update pipeline config to reference the TensorRT engine
# Then run DeepStream pipeline
deepstream-app -c configs/deepstream_app_config.txt
```

**What to capture:**
- Screenshot of the video display window showing bounding box overlays and real-time OSD (On-Screen Display) with FPS and latency counters.
- Save as `results/tensorrt_evidence/deepstream_runtime.png`.
