# CPU batched Clifford output-reuse scorecard

## Decision

Pass for the measured 18-qubit, batch-32, complex128 CPU inference scope.
FlagQuantum now alternates two owned state buffers across native Clifford
matching regions when every intervening region is an in-place native one-qubit
layer. The final state is bitwise identical to the rollback path. Median
execution RSS growth falls by 2.7%, warm time improves by 1.117x, and
FlagQuantum completes the same task 1.327x faster than PennyLane
Lightning's native broadcast batch.

Programs containing a functional dense region, including the current local
brickwork corpus, deliberately retain the previous allocation path. Keeping a
scratch state alive across those regions reduced time but raised peak RSS, so
that broader experiment was rejected rather than shipped.

## What the benchmark measures

The task evaluates 32 independent parameter bindings of the same 18-qubit
Random Clifford circuit and returns all 32 exact complex128 statevectors. Each
statevector contains 262,144 amplitudes; the logical output is 128 MiB. The
circuit contains 126 gates per item and 576 independently supplied parameter
values across the batch.

Warm time is the median of 11 complete calls after two warmups. Peak RSS is the
median of three separate cold Python processes, and both JSON artifacts retain
every raw observation. Each engine uses one CPU thread. Circuit construction is
outside warm timing and inside the cold-process RSS measurement.

## Exact rollback result

Both rows use the same source tree and native kernels. The rollback row sets
`FQ_CPU_CLIFFORD_MATCHING_OUTPUT_REUSE=0`, which restores one fresh output
allocation per matching region.

| Path | Median time | Execution RSS growth | Peak RSS | Relative result |
| --- | ---: | ---: | ---: | --- |
| Reused owned output | 275.685 ms | 389.1 MiB | 569.2 MiB | selected |
| Exact allocation rollback | 307.902 ms | 399.8 MiB | 592.2 MiB | 1.117x slower, 2.7% more execution RSS growth |

The accepted result is a 32.217 ms timing reduction together with a 10.7 MiB
reduction in median execution RSS growth and 23.0 MiB reduction in median total
peak RSS. The memory reduction is modest; the main measured benefit is avoiding
repeated full-state allocations on the warm execution path.

## Prominent PennyLane Lightning comparison

PennyLane Lightning uses `lightning.qubit`, analytic state output, complex128,
one CPU thread, and public broadcast expansion with one-time preprocessing
outside warm timing. Both engines return the same 32 exact statevectors;
maximum cross-framework absolute error is
`1.3438229978002476e-16`.

| Engine | Median time | Relative speed | Peak RSS | Execution RSS growth |
| --- | ---: | ---: | ---: | ---: |
| **FlagQuantum native batch** | **275.685 ms** | **1.327x faster** | 569.2 MiB | 389.1 MiB |
| PennyLane Lightning native batch | 365.846 ms | 1.000x | **543.6 MiB** | **288.7 MiB** |

FlagQuantum is faster for this measured task, while PennyLane Lightning uses
4.7% less total peak RSS and 25.8% less execution RSS growth. The result is
local comparison evidence, not a universal framework ranking or a
scalability/release claim.

## Implementation and safety

The native Clifford kernel now accepts an optional non-aliasing output tensor.
The statevector executor only reuses a previous owned state when the complete
compiled program contains native Clifford matchings and in-place native
one-qubit layers. The initial input is never reused as scratch, caller-owned
inputs remain unchanged, autograd keeps its functional path, and shape, dtype,
device, contiguity, and non-aliasing checks remain at the native dispatch
boundary.

The focused test executes three matching regions, proves that reuse starts only
after two owned buffers exist, compares bitwise with the environment rollback,
and retains the existing complex64, complex128, input-preservation, and
autograd checks.

## Reproduce

Install the local package and comparison dependency, then run the selected path:

```bash
pip install -e '.[pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch pennylane_lightning_native_batch \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-clifford-reuse.json \
  --markdown-output BATCHED_CLIFFORD_REUSE.md
```

Run the exact FlagQuantum rollback with the same profile:

```bash
FQ_CPU_CLIFFORD_MATCHING_OUTPUT_REUSE=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-clifford-reuse-rollback.json
```

The checked raw results are
[`batched_statevector_clifford_output_reuse_cpu_arm64_20261002.json`](batched_statevector_clifford_output_reuse_cpu_arm64_20261002.json)
and
[`batched_statevector_clifford_output_reuse_rollback_cpu_arm64_20261002.json`](batched_statevector_clifford_output_reuse_rollback_cpu_arm64_20261002.json).

## Boundaries and next target

- This is inference-only CPU statevector work; it does not improve backward or
  adjoint execution.
- Reuse is intentionally refused when a functional dense region could overlap
  the retained scratch with a larger temporary working set.
- Peak RSS includes interpreter, framework, circuit construction, and allocator
  behavior. Execution RSS growth subtracts the fresh process's pre-execution
  high-water mark but is still host-local evidence.
- The next memory target remains final chunk assembly or a bounded scratch
  policy for functional dense regions that proves both memory and throughput
  safety before admission.
