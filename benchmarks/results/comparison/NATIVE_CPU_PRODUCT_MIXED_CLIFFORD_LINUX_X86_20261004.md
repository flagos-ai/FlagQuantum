# Native CPU product-state mixed-Clifford matching on Linux x86-64

This report records the measured effect of routing wide product-state components
through FlagQuantum's existing native mixed-`CX`/`CZ` matching kernel. Raw samples
are in
[`random_clifford_product_mixed_linux_x86_20261004.json`](random_clifford_product_mixed_linux_x86_20261004.json);
the exact rollback samples are in
[`random_clifford_product_mixed_rollback_linux_x86_20261004.json`](random_clifford_product_mixed_rollback_linux_x86_20261004.json).

## What this measures and why it matters

The deterministic 22-qubit complex128 Random Clifford circuit has four layers,
132 gates, logical depth 8, 21 `CX` gates and 23 `CZ` gates. Its shuffled
two-qubit matchings have a mean wire span of 8.5, so this is a useful stress case
for non-local statevector traffic rather than a nearest-neighbour-only microbenchmark.

The product-state executor used to expand each wide mixed matching into separate
full-state `CZ` scans and `CX` gathers after the initially independent components
became entangled. The new route keeps small components on the existing product
path, but sends a static parameter-free mixed matching in a component of at least
16 wires through one portable native C++/ATen pass.

## Measured result

Measurements used one socket of an Intel Xeon Platinum 8358 host: CPUs `0-31`,
32 physical cores, two warmups and seven retained end-to-end samples. Conversion,
backend preparation, execution and statevector retrieval are included. Engine
order rotated between retained samples.

| Engine or path | Median | RMAD | Relative to optimized FlagQuantum |
| --- | ---: | ---: | ---: |
| **FlagQuantum optimized** | **50.279 ms** | 2.06% | **1.00×** |
| FlagQuantum exact rollback | 162.768 ms | 0.62% | 3.24× slower |
| Qiskit Aer 0.17.2 | 189.127 ms | 3.33% | **FlagQuantum 3.76× faster** |
| PennyLane Lightning 0.45.0 | 467.867 ms | 1.65% | **FlagQuantum 9.31× faster** |

All timing groups met the declared 20% RMAD stability threshold. The optimized
path is 3.24× faster than the exact rollback and is now the fastest of the three
matched public execution paths on this host.

## Correctness and route evidence

- Optimized versus exact rollback maximum absolute statevector error: `0.0`.
- Qiskit Aer maximum absolute error versus FlagQuantum: `8.67e-18`.
- PennyLane Lightning maximum absolute error versus FlagQuantum: `9.11e-18`.
- The optimized run recorded two `native_cpu_clifford_matching_regions`; the
  rollback recorded zero.
- Linux x86 focused coverage passed: 28 tests covering product-state routing,
  cache isolation, native execution, fallback and rollback.

## Public usage

There is no new user-facing backend option. Normal FlagQuantum code selects the
route automatically:

```python
import random

import flagquantum as fq
import torch

n_qubits = 22
generator = random.Random(7319 + 10_007 * n_qubits)
circuit = fq.Circuit(n_qubits, dtype=torch.complex128)
for _ in range(4):
    for qubit in range(n_qubits):
        getattr(circuit, generator.choice(("h", "s", "x")))(qubit)
    qubits = list(range(n_qubits))
    generator.shuffle(qubits)
    for index in range(0, n_qubits - 1, 2):
        getattr(circuit, generator.choice(("cx", "cz")))(
            qubits[index], qubits[index + 1]
        )

state = circuit.state()
```

`FQ_CPU_NATIVE_CLIFFORD_MATCHING=0` is the exact rollback switch. It restores
the previous product-state program and is also part of the compiled-program cache
key, so enabled and rollback programs cannot be confused in one process.

## Boundaries

- The checked-in performance claim is limited to this 22-qubit complex128,
  parameter-free exact-statevector workload on the stated Linux x86 host.
- The optimization applies only on CPU when the optional native extension is
  available, the product-state executor is selected, a matching contains `CX`,
  and the affected component spans at least 16 wires.
- Small components, unsupported gates, custom matrices, parameters, batches,
  non-CPU devices and unavailable extensions retain the established path.
- This is forward-simulation evidence, not a gradient, release-scalability or
  universal framework-ranking claim.

## Reproduction

From the repository root, with Qiskit Aer and PennyLane Lightning installed:

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
taskset -c 0-31 flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector --n-wires 22 \
  --engines flagquantum_native qiskit_aer pennylane_lightning_qubit \
  --threads 32 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --json-output random-clifford-product-mixed-candidate.json

FQ_CPU_NATIVE_CLIFFORD_MATCHING=0 \
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
taskset -c 0-31 flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector --n-wires 22 \
  --engines flagquantum_native \
  --threads 32 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --json-output random-clifford-product-mixed-rollback.json
```

## Stop decision

The change is eligible to merge only when all of the following remain true:

- optimized and rollback statevectors agree within `1e-10`;
- the native route is observed and the rollback bypasses it;
- the optimized median is at least 1.05× faster than rollback;
- optimized FlagQuantum is no slower than the fastest measured external engine;
- focused tests and repository CI pass with no new static-check failures.

This measurement satisfies the first four conditions. Repository CI is the final
gate for the PR.
