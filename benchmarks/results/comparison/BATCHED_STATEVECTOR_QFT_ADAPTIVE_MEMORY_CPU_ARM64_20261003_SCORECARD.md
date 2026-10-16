# CPU batched Truncated QFT adaptive-memory scorecard

## Conclusion

FlagQuantum now gives controlled-phase-graph programs the measured 32 MiB CPU
parameter-batch window. On this Apple-arm64 profile, the 18-qubit, batch-32
Truncated QFT task reduces median execution RSS growth from **320.8 MiB** to
**176.9 MiB**, a **44.9% reduction**, relative to the fixed-64-MiB rollback.
It also completes in **856.183 ms** instead of **906.303 ms**, a **1.059x
speedup**. PennyLane Lightning takes **2085.338 ms** on the matched public
native-broadcast path, so FlagQuantum is **2.436x faster**. All outputs agree
within `1e-10`, and every timing series passes the 20% relative-MAD rule.

## What changed

The adaptive selector already used a 32 MiB logical-state window for measured
CX-sequence, cross-wire-diagonal, and static-CZ-graph programs. This change adds
the compiled controlled-phase graph used by Truncated QFT. The decision is made
from the compiled program, not a workload name. Other program classes keep the
64 MiB general budget.

`FQ_CPU_STATEVECTOR_BATCH_ADAPTIVE_BUDGET=0` remains the complete rollback to
the fixed 64 MiB policy. Runtime metrics continue to expose the selected budget,
chunk size, chunk count, and assembly strategy.

## Measured result

Apple arm64, one CPU thread, Python 3.12.14, PyTorch 2.13.0, PennyLane 0.45.1,
and PennyLane Lightning 0.45.0. Times are medians of 11 calls after three
warmups. RSS growth is the median of three fresh-process `ru_maxrss` probes.

| 18-qubit, batch-32 Truncated QFT | Total time | Relative to FlagQuantum | Execution RSS growth |
| --- | ---: | ---: | ---: |
| **FlagQuantum adaptive 32 MiB** | **856.183 ms** | 1.000x | **176.9 MiB** |
| FlagQuantum fixed-64 rollback | 906.303 ms | **1.059x slower** | 320.8 MiB |
| PennyLane Lightning native broadcast | 2085.338 ms | **2.436x slower** | 256.0 MiB |

The output itself is 128 MiB. The optimization removes 143.9 MiB of measured
execution RSS growth relative to rollback and lowers FlagQuantum below
Lightning's execution growth on this case.

## User code

No API change is required. A parameter batch around a QFT-style phase graph
automatically receives the measured window policy:

```python
import torch
import flagquantum as fq
from flagquantum.algorithms.primitives import append_qft

batch = 32
qubits = 18
angles = torch.linspace(-0.4, 0.4, batch, dtype=torch.float64)

circuit = fq.Circuit(qubits, bsz=batch, dtype=torch.complex128)
append_qft(circuit, list(range(qubits)))
for qubit in range(qubits):
    circuit.ry(qubit, angles + 0.01 * qubit)

states = circuit.state()
assert states.shape == (batch, 2**qubits)
```

## Reproduce

From a source checkout with the PennyLane extra installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads truncated_qft_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_adaptive_budget_rollback \
    pennylane_lightning_native_batch \
  --threads 1 --warmup 3 --iterations 11 --memory-probes 3 \
  --json-output qft-adaptive-memory.json \
  --markdown-output QFT_ADAPTIVE_MEMORY.md
```

The adjacent JSON contains all timing samples, fresh-process RSS probes,
versions, workload hash, correctness errors, and methodology fields.

## Limits and stop condition

- This is single-host Apple-arm64 comparison evidence, not a universal
  scalability or framework claim.
- It covers exact CPU forward statevectors with `complex128`; it does not claim
  backward, shots, GPU, or distributed-memory improvement.
- The corpus uses a range-3 Truncated QFT plus one independent terminal RY angle
  per qubit. It does not claim the same ratio for every QFT depth or batch size.
- RSS is a process high-water mark; the result therefore uses three fresh
  processes and reports their median.
- This PR stops because the remaining QFT target exceeds 35% RSS reduction,
  does not regress runtime, remains at least 2x faster than Lightning, preserves
  exact correctness, and retains a one-variable rollback.
