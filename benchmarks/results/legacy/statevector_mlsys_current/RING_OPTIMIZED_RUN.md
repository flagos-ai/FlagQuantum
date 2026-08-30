# Ring optimized run

Use the same 31-qubit, 8-layer value-and-gradient workload for every device
count. The persistent layout switch is required for both the forward and the
reversible-adjoint backward pass; without it the comparison is the canonical
baseline, not the optimized implementation.

```bash
export FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT=1
export FQ_STATEVECTOR_TRITON_LOCAL_CX=1
export FQ_STATEVECTOR_TRITON_CX_SEGMENT=1
export FQ_STATEVECTOR_CROSS_SHARD_CX_PACK=1

torchrun --nproc-per-node=${NPROC} \
  benchmarks/statevector_training_scaling.py \
  --n-wires=31 --workload=full-width-ring --layers=8 \
  --warmup=2 --repetitions=5 \
  --enable-triton-vjp-adjoint --enable-fused-vjp-pipeline \
  --enable-cross-shard-cx-pack --enable-forward-cross-shard-cx-pack \
  --json-output=benchmarks/results/legacy/statevector_mlsys_current/generality/flagquantum_31q_d8_ring_${NPROC}gpu_optimized.json
```

The optimized path keeps the logical-to-physical permutation persistent across
layers, schedules independent gates before layout swaps, and compiles local CX
segments. Results must report value+gradient time, logical communication bytes,
communication kernel count, and rank-max timing; do not mix baseline and
optimized payloads in one scaling curve.
