#!/usr/bin/env bash
# ==============================================================================
# AdaPipe - TensorRT Engine Performance Benchmark Script
# Measures exact raw engine latency, percentiles (p50, p95, p99), and throughput (FPS)
# using trtexec standalone harness.
# ==============================================================================

set -euo pipefail

EVIDENCE_DIR="results/tensorrt_evidence"
ENGINE_FP16="${EVIDENCE_DIR}/model_fp16.engine"
ENGINE_FP32="${EVIDENCE_DIR}/model_fp32.engine"

WARMUP_MS=5000
DURATION_SEC=45
ITERATIONS=1000

echo "========================================================"
echo " AdaPipe TensorRT Performance Benchmarking"
echo " Protocol: Sequential execution, 5s warmup, identical input shape"
echo "========================================================"

if ! command -v trtexec &> /dev/null; then
    echo "ERROR: 'trtexec' executable not found in PATH."
    exit 1
fi

# Benchmark FP16 Engine
if [ -f "${ENGINE_FP16}" ]; then
    echo "--> Running FP16 Benchmark..."
    trtexec \
        --loadEngine="${ENGINE_FP16}" \
        --warmUp="${WARMUP_MS}" \
        --duration="${DURATION_SEC}" \
        --iterations="${ITERATIONS}" \
        --percentile=50,95,99 \
        --dumpProfile \
        --exportTimes="${EVIDENCE_DIR}/benchmark_times_fp16.json" \
        > "${EVIDENCE_DIR}/benchmark_fp16.log" 2>&1
    echo "FP16 benchmark complete. Log: ${EVIDENCE_DIR}/benchmark_fp16.log"
else
    echo "Notice: ${ENGINE_FP16} not found. Run compile_tensorrt.sh first."
fi

# Benchmark FP32 Engine
if [ -f "${ENGINE_FP32}" ]; then
    echo "--> Running FP32 Benchmark..."
    trtexec \
        --loadEngine="${ENGINE_FP32}" \
        --warmUp="${WARMUP_MS}" \
        --duration="${DURATION_SEC}" \
        --iterations="${ITERATIONS}" \
        --percentile=50,95,99 \
        --dumpProfile \
        --exportTimes="${EVIDENCE_DIR}/benchmark_times_fp32.json" \
        > "${EVIDENCE_DIR}/benchmark_fp32.log" 2>&1
    echo "FP32 benchmark complete. Log: ${EVIDENCE_DIR}/benchmark_fp32.log"
else
    echo "Notice: ${ENGINE_FP32} not found. Run compile_tensorrt.sh first."
fi
