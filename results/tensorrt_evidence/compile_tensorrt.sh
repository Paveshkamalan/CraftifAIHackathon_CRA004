#!/usr/bin/env bash
# ==============================================================================
# AdaPipe - TensorRT Engine Compilation Script for NVIDIA Target Platforms
# Supported Targets: NVIDIA RTX 3050, Jetson Orin, or Organizer Cloud GPU
# Frameworks: TensorRT 8.6 / 10.x, CUDA 12.x / 13.x, DeepStream 6.3 / 7.1
# ==============================================================================

set -euo pipefail

MODEL_ONNX="pipegen_output/generated_pipeline/model.onnx"
OUTPUT_DIR="results/tensorrt_evidence"
mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo " AdaPipe TensorRT Engine Compilation"
echo " Target ONNX Model: ${MODEL_ONNX}"
echo "========================================================"

# Check for trtexec
if ! command -v trtexec &> /dev/null; then
    echo "ERROR: 'trtexec' executable not found in PATH."
    echo "Please ensure NVIDIA TensorRT is installed and /usr/src/tensorrt/bin or TensorRT bin directory is in PATH."
    exit 1
fi

# Check for nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: 'nvidia-smi' not found. NVIDIA driver is required."
    exit 1
fi

echo "Detected GPU Architecture:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ------------------------------------------------------------------------------
# 1. Compile FP32 TensorRT Engine
# ------------------------------------------------------------------------------
echo ""
echo "--> Compiling FP32 TensorRT Engine..."
trtexec \
    --onnx="${MODEL_ONNX}" \
    --saveEngine="${OUTPUT_DIR}/model_fp32.engine" \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOnly \
    --verbose \
    > "${OUTPUT_DIR}/trtexec_compile_fp32.log" 2>&1

echo "FP32 compilation complete. Engine saved to ${OUTPUT_DIR}/model_fp32.engine"

# ------------------------------------------------------------------------------
# 2. Compile FP16 TensorRT Engine (Default PipeGen Target)
# ------------------------------------------------------------------------------
echo ""
echo "--> Compiling FP16 TensorRT Engine..."
trtexec \
    --onnx="${MODEL_ONNX}" \
    --saveEngine="${OUTPUT_DIR}/model_fp16.engine" \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --buildOnly \
    --verbose \
    > "${OUTPUT_DIR}/trtexec_compile_fp16.log" 2>&1

echo "FP16 compilation complete. Engine saved to ${OUTPUT_DIR}/model_fp16.engine"

# ------------------------------------------------------------------------------
# 3. Compile INT8 TensorRT Engine (Requires Calibration Cache)
# ------------------------------------------------------------------------------
echo ""
echo "--> Note on INT8 Compilation:"
echo "INT8 quantization requires dynamic range calibration using representative dataset frames."
echo "If calibration cache (calibration.cache) is available, run:"
echo "trtexec --onnx=${MODEL_ONNX} --saveEngine=${OUTPUT_DIR}/model_int8.engine --int8 --calib=calibration.cache"
echo ""

echo "Compilation process finished."
