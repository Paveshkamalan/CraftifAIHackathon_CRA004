# AdaPipe - Run Candidate Benchmarking (PowerShell)
param(
    [string]$Config = "configs/slo.yaml",
    [double]$Duration = 45.0,
    [double]$Warmup = 5.0
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  AdaPipe: Running Candidate Benchmarks" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

python -m adapipe.main --mode benchmark --config $Config --duration $Duration --warmup $Warmup
