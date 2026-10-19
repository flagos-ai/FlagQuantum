# Static Clifford layer CPU scorecard

## Conclusion

For the measured 18-qubit, batch-32 Random Clifford task, the specialized
FlagQuantum kernel reduces the median complete-task time from **572.322 ms to
289.961 ms**, a **1.974x speedup** over its focused rollback. It is also **1.468x
faster than the measured PennyLane Lightning bridge** at 425.728 ms. All eleven
warm timing samples pass the stability rule and the maximum cross-engine state
error is 1.35e-16.

This closes the previously measured Lightning gap for this exact workload and
machine. It is not a claim that FlagQuantum is faster for every circuit: the
optimization applies only to CPU inference batches containing disjoint static
H/S/Sdg/X/Y/Z layers.

## What is measured and why it matters

The task returns 32 exact complex128 statevectors for independent parameter
bindings of one deterministic 18-qubit circuit. The Random Clifford case has
126 gates per item and 576 independent parameters across the batch. Complete
execution is timed after construction and warmup, including result retrieval.
This is the user-facing batch task rather than an isolated kernel microbenchmark.

Random Clifford repeatedly exposes full-width static single-qubit layers. The
old native path represented every gate as a dense 2x2 matrix, split an 18-wire
layer into bounded blocks, and traversed the state once per block. The new C++
kernel aggregates X/Y permutations and S/Sdg/Y/Z phases into bit masks, then
performs a Walsh-Hadamard transform only over H wires. It executes the full
disjoint layer in one state traversal while keeping the existing dense native
kernel as the rollback path.

The local-brickwork control contains RY/RZ and CX gates but no eligible static
Clifford layer. Therefore its optimized and rollback executions are identical;
their observed 976.444 ms and 890.106 ms medians are run-order noise and must not
be attributed to this change. PennyLane Lightning measures 995.795 ms on that
control in the same run.

## Results

| Workload | Engine | Median task time | Relative result | Peak RSS | Maximum state error |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Clifford | **FlagQuantum optimized** | **289.961 ms** | **1.974x faster than rollback** | 592.7 MiB | 0 |
| Random Clifford | FlagQuantum focused rollback | 572.322 ms | baseline | 594.4 MiB | 1.35e-16 |
| Random Clifford | **PennyLane Lightning bridge** | **425.728 ms** | **FlagQuantum is 1.468x faster** | 535.4 MiB | 1.35e-16 |
| Local brickwork control | FlagQuantum optimized | 976.444 ms | not applicable | 840.9 MiB | 0 |
| Local brickwork control | FlagQuantum focused rollback | 890.106 ms | same executed path | 793.5 MiB | 0 |
| Local brickwork control | PennyLane Lightning bridge | 995.795 ms | 1.020x slower than FQ sample | 520.9 MiB | 5.73e-17 |

Random Clifford peak RSS is effectively unchanged versus rollback (592.7 vs
594.4 MiB), but remains 10.7% above Lightning's 535.4 MiB. Timing and memory
details, all raw samples, package versions, IR hashes, and correctness fields are
retained in the [JSON artifact](batched_statevector_static_clifford_layer_cpu_arm64_20261001.json).
The generated [comparison table](https://github.com/FlagQuantum/FlagQuantum-evidence/releases/tag/evidence-2026-10-09.2)
is the compact view of the same data.

## User code

No public API changes are required. Eligible layers are selected automatically:

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(6, bsz=32, dtype=torch.complex128)
for wire, gate in enumerate(("h", "s", "sdg", "x", "y", "z")):
    getattr(circuit, gate)(wire)

states = circuit.state()  # shape: (32, 2**6)
```

Set `FQ_CPU_NATIVE_STATIC_CLIFFORD_LAYER=0` to retain all other native CPU
optimizations while restoring the former dense-matrix execution for these
layers. Autograd, non-CPU execution, scalar batches, custom matrices, and
parameterized gates continue through their existing paths.

## Reproduction

From a source checkout with the native CPU extension built and PennyLane extras
installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_static_clifford_layer_rollback \
    pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-static-clifford.json \
  --markdown-output BATCHED_STATIC_CLIFFORD.md
```

The result is single-host Apple-arm64 comparison evidence. It is not a release
gate, multi-thread scaling result, distributed result, or universal framework
ranking.
