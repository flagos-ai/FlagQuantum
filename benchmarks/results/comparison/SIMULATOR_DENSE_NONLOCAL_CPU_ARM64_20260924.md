# Dense nonlocal CPU simulator comparison

This report measures an exact 22-qubit statevector circuit containing one
Hadamard layer, the complete graph of 231 CZ gates, and one parameterized RY
layer. The 275-gate workload tests whether a simulator can execute a large
commuting diagonal graph without repeatedly streaming the full 64 MiB
complex128 statevector.

FlagQuantum compiles any consecutive static CZ graph into one exact Boolean
quadratic phase application. At 22 qubits the cached signs occupy 4 MiB as
int8 values. The rollback path partitions the complete graph into 21
wire-disjoint matchings and applies each matching separately.

## Measured result

The run used one CPU thread on Apple arm64 with Python 3.12.14 and PyTorch
2.13.0. Each engine received two warmups followed by seven retained samples,
with seven calls per sample. Conversion, backend preparation, execution, and
statevector retrieval are included. Every engine matched the FlagQuantum
reference within absolute tolerance `1e-10`, and every timing group passed the
20% relative median absolute deviation threshold.

| Engine | Median | RMAD | External / FlagQuantum |
| --- | ---: | ---: | ---: |
| FlagQuantum native | 0.287360 s | 7.35% | 1.00x |
| PennyLane Lightning | 0.347721 s | 6.14% | 1.21x |
| Cirq Simulator | 0.802903 s | 5.94% | 2.79x |
| Qiskit Aer | 1.668638 s | 4.29% | 5.81x |

A separate native-only rollback A/B used two warmups, seven retained samples,
and five calls per sample. At 22 qubits the median fell from 0.928177 seconds
with `FQ_CPU_CZ_GRAPH_FUSION=0` to 0.300677 seconds with the optimization,
a 3.09x speedup. The same A/B measured 2.60x, 2.75x, and 4.34x at 10, 14,
and 18 qubits respectively.

These results establish a local CPU advantage for this exact circuit family;
they are not a universal framework ranking, a distributed scalability claim,
or evidence for noisy, dynamic, sparse-output, or non-statevector workloads.

## User code

```python
import torch

import flagquantum as fq

circuit = fq.Circuit(22, dtype=torch.complex128)
for wire in range(22):
    circuit.h(wire)
for left in range(22):
    for right in range(left + 1, 22):
        circuit.cz(left, right)
for wire in range(22):
    circuit.ry(wire, 0.031 * (wire + 1))

state = circuit.state()
```

No user-facing optimization switch is required. To reproduce the comparison:

```bash
pip install -e '.[qiskit,cirq,pennylane]'
flagquantum-benchmark run simulator_workload_corpus \
  --workloads dense_nonlocal_statevector --n-wires 22 \
  --engines flagquantum_native qiskit_aer cirq_simulator \
    pennylane_lightning_qubit \
  --threads 1 --warmup 2 --iterations 7 --calls-per-sample 7 \
  --json-output dense-nonlocal.json
```

The machine-readable measurements are checked in beside this report as
`simulator_dense_nonlocal_cpu_arm64_20260924.json`.
