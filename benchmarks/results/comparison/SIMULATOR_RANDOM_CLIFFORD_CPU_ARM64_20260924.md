# Random Clifford CPU simulator comparison

This report summarizes the raw artifact
[`simulator_random_clifford_cpu_arm64_20260924.json`](simulator_random_clifford_cpu_arm64_20260924.json).
It measures exact complex128 statevectors for the deterministic Random Clifford
workload on one Apple arm64 CPU thread. The circuit has four alternating layers
of per-wire `H`/`S`/`X` gates and shuffled disjoint `CX`/`CZ` matchings. The
18-qubit case contains 108 gates at logical depth 8; the 22-qubit case contains
132 gates at the same depth.

Each median uses two warmups followed by nine retained samples with five
end-to-end calls per sample. Engine order rotates within one process. Conversion,
preparation, execution, and result retrieval are included. Every engine matched
the FlagQuantum statevector at absolute tolerance `1e-10`, and every timing group
met the declared 20% relative median absolute deviation stability threshold.

| Qubits | FlagQuantum | Qiskit Aer | Cirq | PennyLane Lightning | Aer / FQ | Cirq / FQ | PennyLane / FQ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 11.495 ms | 51.943 ms | 13.293 ms | 11.274 ms | 4.52× | 1.16× | 0.98× |
| 22 | 181.030 ms | 592.915 ms | 209.421 ms | 238.416 ms | 3.28× | 1.16× | 1.32× |

A ratio above one means FlagQuantum was faster. At 18 qubits FlagQuantum is
faster than Cirq and within 2% of PennyLane. At 22 qubits FlagQuantum is the
fastest of the four measured engines. These are reproducible results for this
circuit family and host, not a universal framework ranking.

## What changed

The product-state executor previously sent standalone `H`, `S`, `Sdg`, `Y`, and
`Z` operations through dense gate-matrix construction and batched matrix
multiplication. The new fixed Clifford kernels operate directly on the affected
two amplitude slices. `X`, `CX`, and `CZ` retain their existing permutation or
diagonal paths; routing and public APIs do not change.

A separate rollback A/B on the same host used one thread, one warmup per path,
and 21 retained end-to-end samples with alternating path order. It changed only
`FQ_CPU_PRODUCT_STATE_FIXED_CLIFFORD`:

| Qubits | Dense-matrix rollback | Fixed kernels | Speedup | Rollback RMAD | Fixed RMAD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 13.348 ms | 11.416 ms | 1.17× | 1.27% | 4.08% |
| 22 | 187.996 ms | 179.003 ms | 1.05× | 5.71% | 6.55% |

The checked-in comparison then refreshed only FlagQuantum native. Qiskit Aer,
Cirq, and PennyLane payloads were preserved field-for-field, and the comparison
ratios were recomputed. Parameterized circuits,
custom matrices, non-Clifford gates, custom input states, batches, and routing
decisions retain their established behavior.

Normal user code needs no backend-specific option:

```python
import random

import flagquantum as fq
import torch

n_wires = 18
generator = random.Random(7319 + 10_007 * n_wires)
circuit = fq.Circuit(n_wires, dtype=torch.complex128)
for _ in range(4):
    for wire in range(n_wires):
        getattr(circuit, generator.choice(("h", "s", "x")))(wire)
    wires = list(range(n_wires))
    generator.shuffle(wires)
    for index in range(0, n_wires - 1, 2):
        getattr(circuit, generator.choice(("cx", "cz")))(
            wires[index], wires[index + 1]
        )

state = circuit.state()
```

`FQ_CPU_PRODUCT_STATE_FIXED_CLIFFORD=0` restores the prior product-state gate
application for rollback and A/B measurement. The broader
`FQ_CPU_PRODUCT_STATE_EXECUTION=0` switch still restores dense execution.

## Reproduce the comparison

From the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector --n-wires 18 22 \
  --engines flagquantum_native \
  --threads 1 --warmup 2 --iterations 9 --calls-per-sample 5 \
  --refresh-from benchmarks/results/comparison/simulator_random_clifford_cpu_arm64_20260924.json \
  --json-output benchmarks/results/comparison/simulator_random_clifford_cpu_arm64_20260924.json
```
