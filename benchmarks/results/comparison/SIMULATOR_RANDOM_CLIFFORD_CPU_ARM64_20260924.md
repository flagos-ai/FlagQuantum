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
| 18 | 9.246 ms | 51.943 ms | 13.293 ms | 11.274 ms | 5.62× | 1.44× | 1.22× |
| 22 | 136.246 ms | 592.915 ms | 209.421 ms | 238.416 ms | 4.35× | 1.54× | 1.75× |

A ratio above one means FlagQuantum was faster. FlagQuantum is the fastest of
the four measured engines at both widths, including a 1.22× advantage over
PennyLane at 18 qubits and a 1.75× advantage at 22 qubits. These are
reproducible results for this circuit family and host, not a universal framework
ranking.

## What changed

The product-state executor already had fixed kernels for individual Clifford
gates, but it still scanned a merged component once per gate. The new path
batches a disjoint `H`/`S`/`X` layer per product component: `H` retains its
fixed numerical kernel, while all `S` phases share one broadcast and all `X`
operations share one multi-axis permutation. It also recognizes exact,
wire-disjoint mixed `CX`/`CZ` matchings. Because those edges commute, the
compiler places each kind together and the product-state executor applies five
or more `CX` operations in one cached gather when they land in the same merged
component. Independent components remain independent.

A separate rollback A/B on the same host used one thread, two warmups per path,
and 11 retained end-to-end samples with three calls per sample and alternating
path order. It changed only `FQ_CPU_PRODUCT_STATE_CLIFFORD_MATCHING`:

| Qubits | Per-gate rollback | Batched layers | Speedup | Rollback RMAD | Batched RMAD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 11.825 ms | 9.777 ms | 1.21× | 2.13% | 3.23% |
| 22 | 182.693 ms | 131.582 ms | 1.39× | 2.22% | 3.74% |

The rollback and optimized paths were bitwise equal at 18 qubits. Their maximum
absolute difference at 22 qubits was `1.31e-18`, well below the comparison
tolerance.

The checked-in comparison refreshed only FlagQuantum native. Qiskit Aer, Cirq,
and PennyLane timing payloads were preserved field-for-field, and the comparison
ratios were recomputed. Parameterized circuits, custom matrices, non-Clifford
gates, custom input states, batches, and routing decisions retain their
established behavior.

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

`FQ_CPU_PRODUCT_STATE_CLIFFORD_MATCHING=0` restores the prior per-gate ordering
and execution for rollback and A/B measurement. The existing
`FQ_CPU_PRODUCT_STATE_FIXED_CLIFFORD=0` disables fixed single-gate kernels. The
broader `FQ_CPU_PRODUCT_STATE_EXECUTION=0` switch still restores dense execution.

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

To reproduce a rollback comparison, run the same native-only command once with
`FQ_CPU_PRODUCT_STATE_CLIFFORD_MATCHING=0` and once with it set to `1`, writing
the outputs to different JSON files.
