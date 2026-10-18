#!/usr/bin/env bash
# AdaPipe - Run Full End-to-End Pipeline
set -euo pipefail

echo "=========================================================="
echo "  AdaPipe: End-to-End Execution"
echo "=========================================================="

python -m adapipe.main --mode all "$@"
