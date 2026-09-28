# Truncated QFT CPU deferred-SWAP comparison

This report measures the deterministic exact-statevector Truncated QFT workload
from the simulator corpus. Each QFT target interacts with at most its next three
wires through the portable `RZ-RZ-CX-RZ-CX` controlled-phase decomposition, and
the circuit finishes with a SWAP reversal. At 22 qubits the workload contains
333 gates and has logical depth 167.

The product-state executor previously materialized every SWAP immediately. The
final 22-qubit reversal therefore copied the full state eleven times. The new
path records the logical-wire order without moving amplitudes, then materializes
canonical public statevector order once with a cached gather table. A merge can
still canonicalize a component earlier when later computation requires it. The
existing controlled-phase graph optimization remains unchanged.

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
| 18 | 0.004167 s | 0.015606 s | 0.050151 s | 0.075421 s | 3.74× |
| 22 | 0.086533 s | 0.154574 s | 1.040753 s | 0.940726 s | 1.79× |

Values above one mean FlagQuantum is faster for that measured case. The measured
FlagQuantum median is 1.79 times faster than Cirq at 22 qubits. This is evidence
for this workload and host, not a universal framework ranking.

## Rollback A/B

The isolated native comparison used two warmups, 11 retained samples, three
calls per sample, and alternating path order. It changes only
`FQ_CPU_PRODUCT_STATE_DEFER_SWAP`:

| Qubits | Eager rollback | Deferred SWAP | Speedup | Rollback RMAD | Deferred RMAD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 0.005553 s | 0.004031 s | 1.38× | 1.54% | 3.28% |
| 22 | 0.102015 s | 0.086267 s | 1.18× | 0.92% | 3.61% |

The eager and deferred results were bitwise identical at both widths. A separate
screen of the SWAP-routing corpus changed by less than 0.3% at 18 and 22 qubits.
Tests cover complex64 and complex128 permutation values and gradients, plus a
parameterized QFT path after the logical SWAP reversal.

## Reproduction

Rerun the optimized FlagQuantum measurements with the original corpus method:

```bash
flagquantum-benchmark run simulator_workload_corpus \
  --workloads truncated_qft_statevector --n-wires 18 22 \
  --engines flagquantum_native --threads 1 --warmup 1 \
  --iterations 9 --calls-per-sample 1 \
  --json-output truncated-qft-optimized.json
```

Run the native A/B by repeating the command with `--warmup 2`,
`--iterations 11`, and `--calls-per-sample 3`, first normally and then with:

```bash
FQ_CPU_PRODUCT_STATE_DEFER_SWAP=0 \
flagquantum-benchmark run simulator_workload_corpus \
  --workloads truncated_qft_statevector --n-wires 18 22 \
  --engines flagquantum_native --threads 1 --warmup 2 \
  --iterations 11 --calls-per-sample 3 \
  --json-output truncated-qft-rollback.json
```

The raw cross-framework samples and workload hashes remain in
`simulator_workload_corpus_cpu_arm64_20260924.json`.
