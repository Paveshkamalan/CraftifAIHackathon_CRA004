# AdaPipe - Run End-to-End Orchestrator (PowerShell)
param(
    [string]$Config = "configs/slo.yaml",
    [string]$Mode = "all"
)

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  AdaPipe: Executing Control Layer ($Mode)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

python -m adapipe.main --mode $Mode --config $Config
