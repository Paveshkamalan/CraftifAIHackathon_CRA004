# ==============================================================================
# AdaPipe - TensorRT Engine Compilation Script for Windows NVIDIA Targets
# Target Platforms: Windows 11 with NVIDIA GeForce RTX / RTX Quadro GPU
# Requirements: TensorRT (trtexec.exe) in PATH, CUDA 12.x/13.x, NVIDIA Driver
# ==============================================================================

param(
    [string]$ModelPath = "pipegen_output/generated_pipeline/model.onnx",
    [string]$OutputDir = "results/tensorrt_evidence"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " AdaPipe TensorRT Engine Compilation (PowerShell)" -ForegroundColor Cyan
Write-Host " Target Model: $ModelPath" -ForegroundColor Cyan
Write-Host "========================================================"

if (!(Test-Path $ModelPath)) {
    Write-Error "Model file not found at $ModelPath"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$trtexec = Get-Command trtexec.exe -ErrorAction SilentlyContinue
if (!$trtexec) {
    Write-Warning "trtexec.exe is not found in PATH."
    Write-Warning "Ensure NVIDIA TensorRT is installed and the bin directory is added to PATH."
    exit 1
}

$nvsmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvsmi) {
    Write-Host "GPU Detected:" -ForegroundColor Green
    & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    Write-Warning "nvidia-smi not detected."
}

# 1. FP32 Engine
Write-Host "`n--> Compiling FP32 Engine..." -ForegroundColor Yellow
& trtexec `
    --onnx="$ModelPath" `
    --saveEngine="$OutputDir/model_fp32.engine" `
    --minShapes=images:1x3x640x640 `
    --optShapes=images:1x3x640x640 `
    --maxShapes=images:1x3x640x640 `
    --buildOnly `
    --verbose `
    *> "$OutputDir/trtexec_compile_fp32.log"

Write-Host "FP32 Engine saved to $OutputDir/model_fp32.engine" -ForegroundColor Green

# 2. FP16 Engine
Write-Host "`n--> Compiling FP16 Engine..." -ForegroundColor Yellow
& trtexec `
    --onnx="$ModelPath" `
    --saveEngine="$OutputDir/model_fp16.engine" `
    --fp16 `
    --minShapes=images:1x3x640x640 `
    --optShapes=images:1x3x640x640 `
    --maxShapes=images:1x3x640x640 `
    --buildOnly `
    --verbose `
    *> "$OutputDir/trtexec_compile_fp16.log"

Write-Host "FP16 Engine saved to $OutputDir/model_fp16.engine" -ForegroundColor Green
