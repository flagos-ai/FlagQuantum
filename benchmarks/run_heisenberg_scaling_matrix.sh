#!/usr/bin/env bash
set -euo pipefail

PYTHON_ENV=${PYTHON_ENV:-/srv/quantum-demo/FlagQuantum/.conda-root/envs/flagquantum-dev}
OUTPUT_DIR=${OUTPUT_DIR:-benchmarks/results/heisenberg_vqe}
N_SITES=${N_SITES:-32}
DEPTH=${DEPTH:-2}
MAX_BOND=${MAX_BOND:-64}
INITIAL_BOND=${INITIAL_BOND:-1}
CHECKPOINT_BUDGET_GIB=${CHECKPOINT_BUDGET_GIB:-2}
FACTORIZATION_BUDGET_GIB=${FACTORIZATION_BUDGET_GIB:-1}
mkdir -p "$OUTPUT_DIR"

for gpu_count in 1 2 4 8; do
  "$PYTHON_ENV/bin/torchrun" --standalone --nproc-per-node="$gpu_count" \
    benchmarks/distributed_mps_heisenberg_vqe.py \
    --n-sites "$N_SITES" --depth "$DEPTH" --max-bond "$MAX_BOND" --steps 1 \
    --cutoff 1e-8 --learning-rate 0.002 --gradient-tolerance 1000 \
    --checkpoint-budget-gib "$CHECKPOINT_BUDGET_GIB" --factorization-budget-gib "$FACTORIZATION_BUDGET_GIB" \
    --save-two-site-factorizations --initial-state random_isometric \
    --initial-bond "$INITIAL_BOND" --exact-reference-max-sites 0 \
    --output "$OUTPUT_DIR/scaling_valid_n${N_SITES}_p${DEPTH}_chi${MAX_BOND}_g${gpu_count}.json"
done
