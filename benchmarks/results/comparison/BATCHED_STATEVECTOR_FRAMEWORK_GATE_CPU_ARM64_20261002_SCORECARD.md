# CPU batched framework-floor regression gate

## Decision

Pass for the measured Apple-arm64, one-thread, 18-qubit, batch-32,
complex128 profile. The CPU regression gate can now preserve both an internal
FlagQuantum baseline and an explicit cross-framework performance floor.

For Random Clifford, FlagQuantum completed the task in **544.188 ms** versus
**941.428 ms** for the PennyLane Lightning bridge, so FlagQuantum was
**1.730x faster**. For Local brickwork, FlagQuantum took **1084.115 ms** versus
**1334.363 ms**, a **1.231x advantage**. The checked gate requires
`native / Lightning <= 0.90`, so a future result must retain at least a 1.111x
lead rather than merely avoid a historical self-regression.

## What the gate protects

Each task returns 32 exact complex128 statevectors for independent parameter
bindings of the same 18-qubit circuit. The baseline contains 11 warm timings
after two warmups and three fresh-process RSS probes for each engine and
workload. Circuit construction is outside warm timing; bridge conversion and
result retrieval are included. External bridges execute 32 public single-item
calls, which is the task exposed through FlagQuantum today, not PennyLane's
best private batching API.

| Workload | FlagQuantum | PennyLane Lightning | FQ speedup | FQ peak RSS | Lightning peak RSS | Native/Lightning ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Clifford | **544.188 ms** | 941.428 ms | **1.730x** | **575.7 MiB** | 600.1 MiB | **0.578** (limit 0.900) |
| Local brickwork | **1084.115 ms** | 1334.363 ms | **1.231x** | **540.1 MiB** | 582.4 MiB | **0.812** (limit 0.900) |

Both engines pass the `1e-10` statevector error contract on both workloads.
All four timing distributions pass the 20% relative-MAD stability rule. The
machine-readable gate verdict is
[`batched_statevector_framework_gate_cpu_arm64_20261002.json`](batched_statevector_framework_gate_cpu_arm64_20261002.json),
and all raw observations remain in the
[`baseline artifact`](batched_statevector_framework_gate_baseline_cpu_arm64_20261002.json).

## User-facing command

Generate a candidate with the same profile:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output candidate-batched-framework.json
```

Evaluate timing, RSS, correctness, stability, completeness, and the Lightning
floor together:

```bash
flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/batched_statevector_framework_gate_baseline_cpu_arm64_20261002.json \
  candidate-batched-framework.json \
  --max-slowdown 1.10 --max-memory-growth 1.10 \
  --minimum-samples 5 --minimum-memory-probes 3 \
  --comparison-engine pennylane_lightning_bridge \
  --max-native-over-comparison 0.90 \
  --json-output candidate-batched-framework-gate.json
```

Exit status `0` is pass, `1` is regression, and `2` means the runtime or
measurement profile is incomparable. The gate also fails closed when the
comparison engine, its correctness result, stable timing, or minimum sample
count is absent.

## Limits

- This is local comparison evidence, not a release, scalability, or universal
  simulator-ranking claim.
- The floor covers only these two workload hashes and this exact runtime
  profile; other circuits and widths require their own evidence.
- The 10% framework margin is intentionally weaker than the measured lead so
  ordinary host noise does not erase the gate's usefulness.
- This PR adds regression governance; it does not claim a new kernel speedup.
