# Batched static-product initialization CPU scorecard

## Conclusion

Pass for the measured 18-qubit, batch-32, complex128 CPU inference scope.
FlagQuantum now initializes a complete leading `H`/`S`/`Sdg`/`X`/`Y`/`Z`
product layer directly in each bounded result slice. The exact rollback takes
**470.059 ms** and the selected path takes **357.607 ms**, a **1.314x speedup**.
Median execution RSS growth falls from **332.8 MiB to 208.0 MiB**, a **37.5%
reduction**.

In the same run, PennyLane Lightning native batch takes **690.560 ms**, so
FlagQuantum is **1.931x faster** for this complete task. FlagQuantum also uses
**40.2% less execution RSS growth** than Lightning's measured **347.6 MiB**.
These are workload-specific Apple-arm64 results, not a universal framework
ranking.

## What is measured and why it matters

The Random Clifford workload has four layers. Each layer applies one randomly
selected `H`, `S`, or `X` gate to all 18 qubits and then a shuffled disjoint
matching of `CX` and `CZ` gates. A terminal `RY` layer carries 576 independent
parameter values across 32 circuit items. Each item has 126 gates and returns
an exact complex128 statevector. The logical output alone is 128 MiB.

Before this change, each bounded window materialized a separate zero state,
ran the first static layer over the full state, and copied the completed window
into the final output. The specialized initializer derives the leading product
state analytically: Hadamards define its support, `X`/`Y` define fixed-one bits,
and `Y` contributes the exact power-of-`i` phase. It zeros the owned final slice
once and writes only the supported amplitudes. The subsequent native Clifford
matchings reuse two buffers, so an even matching count returns the completed
state to the final slice without a last copy.

Warm time is the median of 11 complete calls after three warmups. Peak RSS is
the median of three fresh-process cold probes. Circuit construction is outside
warm timing and inside RSS measurement. Each engine uses one CPU thread.

## Formal results

| Engine | Median task time | Relative result | Peak RSS | Execution RSS growth | Maximum state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| **FlagQuantum static-product direct initialization** | **357.607 ms** | **1.314x faster than rollback** | **387.4 MiB** | **208.0 MiB** | 0 |
| FlagQuantum zero-state/static-layer rollback | 470.059 ms | baseline | 513.7 MiB | 332.8 MiB | 0 |
| **PennyLane Lightning native batch** | **690.560 ms** | **FlagQuantum is 1.931x faster** | 601.1 MiB | 347.6 MiB | 1.34e-16 |

The selected, rollback, and Lightning relative median absolute deviations are
7.42%, 11.07%, and 8.34%; all pass the 20% stability rule. Raw samples,
versions, IR hash, correctness values, and every memory probe are retained in
the [JSON artifact](batched_statevector_static_product_initialization_cpu_arm64_20261003.json).

## User code

No new public API is required. Eligible inference circuits select the native
path automatically:

```python
import torch
import flagquantum as fq

n_qubits = 18
batch_size = 32
circuit = fq.Circuit(n_qubits, bsz=batch_size, dtype=torch.complex128)

for layer in range(4):
    for qubit in range(n_qubits):
        gate = (circuit.h, circuit.s, circuit.x)[(qubit + layer) % 3]
        gate(qubit)
    for left in range(layer % 2, n_qubits - 1, 2):
        (circuit.cx if (left + layer) % 2 == 0 else circuit.cz)(left, left + 1)

angles = torch.linspace(-0.4, 0.4, batch_size, dtype=torch.float64)
for qubit in range(n_qubits):
    circuit.ry(qubit, angles + 0.01 * qubit)

states = circuit.state()  # shape: (32, 2**18)
```

## Implementation and safety

The compiler only selects the initializer when consecutive leading native
static-Clifford tiles cover every qubit exactly once. The complete remaining
program must consist of native fixed or parameterized layers and disjoint
`CX`/`CZ` matchings before those bounded windows may occupy final-result slices.

The path is CPU-inference-only. Autograd, caller-provided initial states,
non-CPU execution, incomplete first layers, custom matrices, and mixed programs
retain the existing path. Focused tests cover complex64 and complex128, reversed
qubit order, every supported fixed Clifford gate, `CX` and `CZ`, batch-specific
terminal parameters, exact rollback, and final-slice ownership. Set
`FQ_CPU_NATIVE_STATIC_PRODUCT_STATE_INITIALIZATION=0` for the complete feature
rollback.

## Reproduce

```bash
pip install -e '.[pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
            flagquantum_native_static_product_initialization_rollback \
            pennylane_lightning_native_batch \
  --threads 1 --warmup 3 --iterations 11 --memory-probes 3 \
  --json-output static-product-initialization.json \
  --markdown-output STATIC_PRODUCT_INITIALIZATION.md
```

## Boundaries and next target

- The fast path requires a complete leading product layer. A circuit that
  entangles before every qubit is initialized keeps the general executor.
- The result covers exact inference statevectors, not gradients, density
  matrices, MPS, arbitrary matrices, or GPU execution.
- The measured four-matching workload returns to the owned output buffer.
  Odd matching counts remain correct but can require a final copy.
- Peak RSS includes interpreter, framework, circuit construction, and allocator
  behavior. This is local comparison evidence, not release-grade scalability
  evidence.
- The next CPU target is the remaining odd-matching output lifetime and its
  benefit across multiple batch sizes, rather than broadening eligibility
  without measured evidence.
