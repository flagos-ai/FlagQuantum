# CPU batched statevector phase closeout

## Conclusion

The measured CPU parameter-batch phase meets its stop condition on this local
profile. FlagQuantum completes all five 18-qubit, batch-32, complex128 exact
statevector tasks faster than PennyLane Lightning's public native broadcast
path, by **1.323x to 2.571x**. All outputs agree within `1e-10`, all timing
series pass the 20% relative-MAD stability rule, and the result includes three
fresh-process RSS probes per engine and workload.

This closes the current single-thread parameter-batched forward phase. It does
not close CPU optimization generally: multi-thread scaling, adjoint gradients,
larger statevectors, and memory-bound workloads remain separate measured
tracks.

## What is measured and why it matters

Each task evaluates 32 independent parameter bindings of one circuit structure
and returns all 32 exact statevectors. Circuit construction and one-time backend
preprocessing are outside warm timing; execution, result retrieval, and result
materialization are included. The five workloads cover layered variational
circuits, controlled phases and routing, shuffled Clifford layers, local
brickwork, and dense nonlocal diagonal structure.

This measures the common workflow in which users sweep parameters without
rewriting their FlagQuantum program. PennyLane Lightning receives the same
logical circuits through its public broadcasted analytic-state interface, not
32 repeated ecosystem bridge calls.

## Measured performance

Apple arm64, one CPU thread, Python 3.12.14, PyTorch 2.13.0, PennyLane 0.45.1,
and PennyLane Lightning 0.45.0. Timings are medians of 11 calls after three
warmups.

| Workload | FlagQuantum | PennyLane Lightning | FlagQuantum speedup | FQ execution RSS growth | Lightning execution RSS growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hardware efficient | **715.373 ms** | 975.289 ms | **1.363x** | 396.9 MiB | **267.2 MiB** |
| Truncated QFT | **724.221 ms** | 1862.037 ms | **2.571x** | 333.8 MiB | **253.2 MiB** |
| Random Clifford | **321.046 ms** | 595.749 ms | **1.856x** | **208.5 MiB** | 298.8 MiB |
| Local brickwork | **960.035 ms** | 1269.888 ms | **1.323x** | **241.2 MiB** | 248.0 MiB |
| Dense nonlocal | **364.847 ms** | 677.154 ms | **1.856x** | 319.2 MiB | **282.0 MiB** |

The smallest timing margin is Local brickwork at 1.323x. FlagQuantum also has
lower execution RSS growth for Random Clifford and Local brickwork. Hardware
efficient, Truncated QFT, and Dense nonlocal remain memory targets even though
their measured execution is faster.

## User code

No backend-specific API is required:

```python
import torch
import flagquantum as fq

angles = torch.linspace(-0.4, 0.4, 32, dtype=torch.float64)
circuit = fq.Circuit(18, bsz=32, dtype=torch.complex128)
for wire in range(18):
    circuit.ry(wire, angles + 0.01 * wire)
for left in range(0, 17, 2):
    circuit.cx(left, left + 1)

states = circuit.state()
assert states.shape == (32, 2**18)
```

## Rejected final-copy candidates

The phase closeout also tested the remaining odd-Clifford-matching final-copy
case before recording this matrix. Moving product-state initialization to the
alternate buffer made batch 32 and 64 slower. A compressed in-place
permutation-cycle implementation was neutral at batch 32 (`1.005x`) and slower
at batch 64 (`0.916x`) than the existing continuous-write output kernel. Both
candidates were removed rather than promoted on allocation-count reasoning.

The existing final copy is therefore retained until a future implementation
demonstrates a stable end-to-end win. This is an explicit measured stop, not a
claim that the copy is theoretically optimal.

## Reproduce

From a source checkout with the PennyLane extra installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch pennylane_lightning_native_batch \
  --threads 1 --warmup 3 --iterations 11 --memory-probes 3 \
  --json-output cpu-phase-closeout.json \
  --markdown-output CPU_PHASE_CLOSEOUT.md
```

The adjacent JSON retains every timing sample, RSS probe, workload hash,
version, correctness error, and methodology field.

## Stop condition and remaining work

This phase stops because the finite forward corpus is correct and stable,
FlagQuantum leads Lightning by at least 1.323x in every measured case, and two
plausible remaining copy-elimination designs failed end-to-end A/B testing.

The next CPU work should start only from a distinct measured gap:

1. reduce execution RSS growth on Hardware efficient, Truncated QFT, and Dense
   nonlocal without losing the timing lead;
2. improve multi-thread scaling independently of this one-thread profile;
3. continue adjoint/backward work under its own gradient corpus; or
4. extend capacity measurements to larger qubit counts without presenting a
   batch-32 result as large-state evidence.

## Limits

- This is one Apple-arm64 host and one software/version profile.
- It covers exact CPU forward statevectors only, not gradients, shots, GPU, or
  distributed execution.
- The 18-qubit, batch-32 logical output is 128 MiB. It is a throughput and
  working-set test, not evidence for 22- or 26-qubit capacity.
- Process peak RSS includes different framework baselines; execution RSS growth
  is shown separately.
- The artifact is local comparison evidence and does not support a universal
  framework ranking or a scalability/release claim.
