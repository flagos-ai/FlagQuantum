#!/usr/bin/env bash
set -euo pipefail

PYTHON_ENV=${PYTHON_ENV:-/srv/quantum-demo/FlagQuantum/.conda-root/envs/flagquantum-dev}
OUTPUT_DIR=${OUTPUT_DIR:-benchmarks/results/heisenberg_vqe}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
N_SITES=${N_SITES:-32}
DEPTH=${DEPTH:-4}
STEPS=${STEPS:-3}
INITIAL_BOND=${INITIAL_BOND:-1}
CHECKPOINT_BUDGET_GIB=${CHECKPOINT_BUDGET_GIB:-2}
FACTORIZATION_BUDGET_GIB=${FACTORIZATION_BUDGET_GIB:-1}
SVD_DRIVER=${SVD_DRIVER:-gesvd}
mkdir -p "$OUTPUT_DIR"

for chi in 128 256 512 768; do
  PYTHONPATH="${PYTHONPATH:-FlagQuantum}" "$PYTHON_ENV/bin/torchrun" \
    --standalone --nproc-per-node=1 \
    benchmarks/distributed_mps_heisenberg_vqe.py \
    --n-sites "$N_SITES" --depth "$DEPTH" --max-bond "$chi" --steps "$STEPS" \
    --cutoff 1e-8 --gradient-tolerance 1000 \
    --checkpoint-budget-gib "$CHECKPOINT_BUDGET_GIB" \
    --factorization-budget-gib "$FACTORIZATION_BUDGET_GIB" \
    --save-two-site-factorizations --initial-state random_isometric \
    --initial-bond "$INITIAL_BOND" --exact-reference-max-sites 20 \
    --svd-driver "$SVD_DRIVER" \
    --output "$OUTPUT_DIR/chi_sweep_n${N_SITES}_p${DEPTH}_chi${chi}.json"
done
