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
| 18 | 13.538 ms | 51.943 ms | 13.293 ms | 11.274 ms | 3.84× | 0.98× | 0.83× |
| 22 | 198.767 ms | 592.915 ms | 209.421 ms | 238.416 ms | 2.98× | 1.05× | 1.20× |

A ratio above one means FlagQuantum was faster. At 18 qubits FlagQuantum is
within 1.8% of Cirq, while PennyLane remains 20% faster. At 22 qubits
FlagQuantum is the fastest of the four measured engines. These are reproducible
results for this circuit family and host, not a universal framework ranking.

## What changed

The existing product-state executor selects a plan from a static estimate of
component work versus dense statevector work. The general limit remains 0.30.
This change permits a measured 0.36 limit only when every compiled operation is
a parameter-free, built-in Clifford gate and no custom matrix is present.

The official 18- and 20-qubit workloads have estimated ratios of 0.350 and
0.317, respectively, so the old general limit sent them to the dense path. A
separate alternating rollback A/B on the same host used one thread, three
warmups, nine retained samples, and seven calls per sample at 18 qubits or three
at 20 qubits:

| Qubits | Dense rollback | Product state | Speedup | Dense RMAD | Product RMAD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 33.034 ms | 16.869 ms | 1.96× | 6.50% | 7.10% |
| 20 | 155.816 ms | 52.392 ms | 2.97× | 1.71% | 3.95% |

The 22-qubit estimate is 0.290, so it already selected product-state execution
before this change and is intentionally unaffected. Parameterized circuits,
custom matrices, non-Clifford gates, custom input states, batches, and plans
above the measured limit retain their established routing.

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

`FQ_CPU_PRODUCT_STATE_EXECUTION=0` restores dense execution for rollback and A/B
measurement.

## Reproduce the comparison

From the repository root:

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector --n-wires 18 22 \
  --threads 1 --warmup 2 --iterations 9 --calls-per-sample 5 \
  --json-output benchmarks/results/comparison/simulator_random_clifford_cpu_arm64_20260924.json
```
