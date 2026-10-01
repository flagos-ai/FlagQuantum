# CPU batched statevector regression-gate scorecard

## Decision

Pass for the measured Apple-arm64, 18-qubit, batch-32, complex128 CPU profile.
The gate reproduces the two #316 allocation hotspots and now fails closed on a
missing case, numerical error, unstable native timing, too few samples, more
than 20% timing slowdown, or more than 10% fresh-process peak-RSS growth.

This PR adds regression governance; it does not claim a new simulator speedup.
The current and baseline revisions use the same statevector kernels, so changes
inside the allowed band are treated as host/run variance.

## Measured gate result

Each current time is the median of 11 complete calls after two warmups. Each
current RSS value is the median of three fresh processes. `Baseline/current`
above one means the current run was faster; `Current/baseline RSS` must remain
at or below `1.10x`.

| Workload | #316 baseline (ms) | Current (ms) | Baseline/current | #316 RSS (MiB) | Current RSS (MiB) | Current/baseline RSS | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random Clifford | 1049.385 | 929.824 | 1.129x | 941.9 | 941.9 | 1.000x | PASS |
| Local brickwork | 1061.349 | 880.708 | 1.205x | 894.5 | 913.8 | 1.022x | PASS |

Both outputs pass exact-statevector parity. Current relative timing MAD is
1.8% for Random Clifford and 3.1% for local brickwork, below the corpus 20%
stability limit. The machine-readable verdict is in
[`batched_statevector_regression_gate_cpu_arm64_20261001.json`](batched_statevector_regression_gate_cpu_arm64_20261001.json),
and every raw timing and RSS observation remains in
[`batched_statevector_regression_current_cpu_arm64_20261001.json`](batched_statevector_regression_current_cpu_arm64_20261001.json).

## Prominent PennyLane Lightning context from #305

The maintained #305 comparison used the same Apple arm64, Python 3.12,
PyTorch 2.13, complex128, one-thread, 18-qubit, batch-32 profile. It is shown
to preserve the external performance target; these values were not rerun and
do not participate in this PR's regression verdict.

| Workload | #305 FlagQuantum time / RSS | PennyLane Lightning time / RSS | Lightning speedup over FQ |
| --- | ---: | ---: | ---: |
| Random Clifford | 1074.345 ms / 1167.6 MiB | 404.360 ms / 671.2 MiB | 2.657x |
| Local brickwork | 934.087 ms / 1042.8 MiB | 863.757 ms / 476.1 MiB | 1.081x |

The bridge comparison executes 32 public single-item Lightning calls, not a
private framework-native batch API. Random Clifford throughput and both
workloads' peak RSS remain explicit optimization targets; a passing regression
gate means they did not materially get worse, not that the gap is closed.

## Reproduce

Generate the current measurement:

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch flagquantum_native_layout_retention \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output candidate-batched-memory.json
```

Evaluate the profile-aware gate:

```bash
flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/batched_statevector_layout_lifetime_cpu_arm64_20261001.json \
  candidate-batched-memory.json --max-slowdown 1.20 \
  --max-memory-growth 1.10 --minimum-samples 5 \
  --minimum-memory-probes 3 \
  --json-output candidate-batched-memory-gate.json
```

Exit status `0` is pass, `1` is regression, and `2` is incomparable. A platform,
Python/PyTorch family, thread profile, timing scope, memory scope, or memory API
mismatch is incomparable instead of a false pass.

## Boundaries

- This is local comparison evidence, not release or scalability evidence.
- Peak RSS includes interpreter/framework startup and circuit construction.
- The gate compares only cases present in the maintained baseline and fails if
  any baseline workload/width/batch/hash identity disappears.
- It does not certify other qubit widths, batch sizes, dtypes, thread counts,
  gradient paths, or external frameworks.
