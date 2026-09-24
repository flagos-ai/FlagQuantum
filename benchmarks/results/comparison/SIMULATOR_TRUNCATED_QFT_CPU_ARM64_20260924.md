# Truncated QFT CPU controlled-phase graph comparison

This report measures the deterministic exact-statevector Truncated QFT workload
from the simulator corpus. Each QFT target interacts with at most its next three
wires through the portable `RZ-RZ-CX-RZ-CX` controlled-phase decomposition, and
the circuit finishes with a SWAP reversal. At 22 qubits the workload contains
333 gates and has logical depth 167.

The optimization recognizes each exact static controlled-phase decomposition,
then combines consecutive decompositions into a weighted phase graph even when
their edges share a wire. The graph's cached complex factors include the exact
global phase of every decomposition. One broadcast multiply replaces several
passes over the current product-state component. Parameterized or approximate
decomposition matches still use the established general path.

## User code

The optimization is automatic; user code does not select a special backend or
rewrite the circuit:

```python
import math

import torch

import flagquantum as fq

circuit = fq.Circuit(22, dtype=torch.complex128)
interaction_range = 3
for target in range(22):
    circuit.h(target)
    for distance in range(1, min(interaction_range, 21 - target) + 1):
        control = target + distance
        theta = math.pi / (2**distance)
        circuit.rz(control, theta / 2)
        circuit.rz(target, theta / 2)
        circuit.cx(control, target)
        circuit.rz(target, -theta / 2)
        circuit.cx(control, target)
for left in range(11):
    circuit.swap(left, 21 - left)

state = circuit.state()
```

## Cross-framework result

The run used one CPU thread on Apple arm64 with Python 3.12.14, PyTorch 2.13.0,
complex128 statevectors, one warmup, and nine retained end-to-end samples. The
FlagQuantum rows were remeasured after this optimization. The external rows are
the unchanged measurements from the checked-in workload corpus, obtained with
the same workload hashes and methodology. All engines matched the exact
reference within absolute tolerance `1e-10`; every timing group passed the 20%
relative median absolute deviation threshold.

| Qubits | FlagQuantum | Cirq | PennyLane | Qiskit Aer | Cirq / FlagQuantum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 0.008044 s | 0.015606 s | 0.050151 s | 0.075421 s | 1.94× |
| 22 | 0.151352 s | 0.154574 s | 1.040753 s | 0.940726 s | 1.02× |

Values above one mean FlagQuantum is faster for that measured case. At 22
qubits the two native engines are close: the measured FlagQuantum median is
2.1% faster than Cirq, not evidence of a universal framework ranking.

## Rollback A/B

The isolated native comparison used two warmups, nine retained samples, and
five calls per sample. It changes only
`FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION`:

| Qubits | Graph enabled | Rollback | Speedup | RMAD enabled | RMAD rollback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 0.007589 s | 0.012042 s | 1.59× | 3.88% | 9.92% |
| 22 | 0.135640 s | 0.198294 s | 1.46× | 1.82% | 1.87% |

The 18-qubit compiled product-state plan uses 44 state applications with the
graph and 75 on rollback. Tests compare graph execution with rollback for
complex64 and complex128, including gradients with respect to the input state.

## Reproduction

Rerun the optimized FlagQuantum measurements with the original corpus method:

```bash
flagquantum-benchmark run simulator_workload_corpus \
  --workloads truncated_qft_statevector --n-wires 18 22 \
  --engines flagquantum_native --threads 1 --warmup 1 \
  --iterations 9 --calls-per-sample 1 \
  --json-output truncated-qft-optimized.json
```

Run the more stable native A/B by repeating the command with `--warmup 2`,
`--calls-per-sample 5`, first normally and then with:

```bash
FQ_CPU_CONTROLLED_PHASE_GRAPH_FUSION=0 \
flagquantum-benchmark run simulator_workload_corpus \
  --workloads truncated_qft_statevector --n-wires 18 22 \
  --engines flagquantum_native --threads 1 --warmup 2 \
  --iterations 9 --calls-per-sample 5 \
  --json-output truncated-qft-rollback.json
```

The raw cross-framework samples and workload hashes remain in
`simulator_workload_corpus_cpu_arm64_20260924.json`.
