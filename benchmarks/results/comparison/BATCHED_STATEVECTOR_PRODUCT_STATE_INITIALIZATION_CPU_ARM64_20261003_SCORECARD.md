# Batched product-state initialization CPU scorecard

## Conclusion

Pass for the measured 18-qubit, batch-32, complex128 CPU inference scope.
FlagQuantum now writes the state produced by the first full rotation/CX layer
directly into bounded slices of the final result. The previous copy-based path
takes **1121.654 ms** and the selected path takes **974.449 ms**, a **1.151x
speedup**. Median execution RSS growth falls from **307.1 MiB to 246.7 MiB**, a
**19.7% reduction**.

In the same run, PennyLane Lightning native batch takes **1291.580 ms**, so
FlagQuantum is **1.325x faster** for this complete task. FlagQuantum also uses
**16.5% less execution RSS growth** than Lightning's measured **295.5 MiB**.
These are workload-specific Apple-arm64 results, not a universal framework
ranking.

## What is measured and why it matters

The local-brickwork workload contains four alternating layers. Every layer has
`RY` and `RZ` on all 18 qubits followed by a nearest-neighbour CX matching; a
final `RY` layer carries 576 independent parameter values across 32 circuit
items. Each item has 196 gates and returns an exact complex128 statevector. The
logical output alone is 128 MiB.

The old bounded-batch path retained a reusable zero-state window, cloned it for
the first native tile, traversed the state for each tile of the first layer, and
then copied completed windows into the final result. The new kernel recognizes
a complete first product layer, forms one- or two-qubit factors (including each
disjoint CNOT), and generates every output amplitude directly in its final
slice. It therefore removes the zero-state clone, two redundant first-layer
state traversals, and the final window copy.

Warm time is the median of 11 complete calls after three warmups. Peak RSS is
the median of three fresh-process cold probes. Circuit construction is outside
warm timing and inside RSS measurement. Each engine uses one CPU thread.

## Formal results

| Engine | Median task time | Relative result | Peak RSS | Execution RSS growth | Maximum state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| **FlagQuantum direct product-state initialization** | **974.449 ms** | **1.151x faster than rollback** | **424.3 MiB** | **246.7 MiB** | 0 |
| FlagQuantum copy-based assembly rollback | 1121.654 ms | baseline | 480.8 MiB | 307.1 MiB | 3.63e-17 |
| **PennyLane Lightning native batch** | **1291.580 ms** | **FlagQuantum is 1.325x faster** | 543.3 MiB | 295.5 MiB | 5.55e-17 |

The selected, rollback, and Lightning relative median absolute deviations are
2.20%, 5.41%, and 7.82%; all pass the 20% stability rule. Raw samples, versions,
IR hash, correctness values, and every memory probe are retained in the
[JSON artifact](batched_statevector_product_state_initialization_cpu_arm64_20261003.json).

## User code

No new public API is required. Eligible inference circuits select the native
path automatically:

```python
import torch
import flagquantum as fq

n_wires = 18
batch_size = 32
circuit = fq.Circuit(n_wires, bsz=batch_size, dtype=torch.complex128)

for layer in range(4):
    for wire in range(n_wires):
        angle = 0.07 * (layer + 1) * (wire + 1)
        circuit.ry(wire, angle)
        circuit.rz(wire, -0.6 * angle)
    for left in range(layer % 2, n_wires - 1, 2):
        circuit.cx(left, left + 1)

angles = torch.linspace(-0.4, 0.4, batch_size, dtype=torch.float64)
for wire in range(n_wires):
    circuit.ry(wire, angles + 0.01 * wire)

states = circuit.state()  # shape: (32, 2**18)
```

## Implementation and safety

The compiler only selects direct initialization when consecutive native
parameterized tiles at the beginning of the program cover every wire exactly
once. Their optional CX edges must already be wire-disjoint. The C++ kernel
precomputes a four-entry factor for each CNOT pair and a two-entry factor for
each unpaired qubit, then streams directly over the final state.

The path is CPU-inference-only and only applies to bounded, initially-zero
parameter batches. Autograd, caller-provided initial states, non-CPU execution,
partial first layers, and mixed programs retain the existing path. Both
complex64 and complex128 are covered by focused numerical tests. Set
`FQ_CPU_STATEVECTOR_BATCH_DIRECT_ASSEMBLY=0` for the complete copy-based
rollback, or `FQ_CPU_NATIVE_PRODUCT_STATE_INITIALIZATION=0` to disable only the
native initializer.

## Reproduce

```bash
pip install -e '.[pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
            flagquantum_native_direct_assembly_rollback \
            pennylane_lightning_native_batch \
  --threads 1 --warmup 3 --iterations 11 --memory-probes 3 \
  --json-output product-state-initialization.json \
  --markdown-output PRODUCT_STATE_INITIALIZATION.md
```

## Boundaries and next target

- The direct path requires a complete initial product layer. Circuits that
  entangle before every wire is initialized keep the general executor.
- The result covers inference statevectors, not gradients, density matrices,
  MPS, arbitrary two-qubit gates, or GPU execution.
- Peak RSS includes interpreter, framework, circuit construction, and allocator
  behavior. This is local comparison evidence, not release-grade scalability
  evidence.
- The next CPU target is to generalize direct initialization to common static
  Clifford product layers without weakening the current eligibility proof.
