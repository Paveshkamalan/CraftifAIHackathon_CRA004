#!/usr/bin/env bash
# AdaPipe - Run Candidate Benchmarking
set -euo pipefail

echo "=========================================================="
echo "  AdaPipe: Running Candidate Benchmarks"
echo "=========================================================="

python -m adapipe.main --mode benchmark "$@"
