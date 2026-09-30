# Native CPU compact CX mapping comparison

This report is generated from
[`native_cpu_adjoint_compact_cx_cpu_arm64_20260930.json`](native_cpu_adjoint_compact_cx_cpu_arm64_20260930.json).
It compares FlagQuantum's compact linear CX mapping with the existing full
permutation-table path and PennyLane Lightning's statevector adjoint. All paths
evaluate the same exact complex128 circuit and complete parameter gradient.

## Results

| Engine | Value evaluation (ms) | Autograd callback (ms) | Value + gradient (ms) | Peak RSS (MiB) | Value + gradient / compact | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FlagQuantum compact CX mapping | 63.190 | 94.187 | 158.922 | 600.766 | 1.000x | 0.000e+00 |
| FlagQuantum full-index rollback | 62.128 | 92.236 | 155.526 | 699.312 | 0.979x | 0.000e+00 |
| PennyLane Lightning adjoint | 1412.456 | 1.188 | 1413.573 | 723.766 | 8.895x | 9.530e-13 |

The framework-comparable metric is value plus gradient. Lightning may perform
adjoint derivative work during value evaluation, so its callback time is not
ranked separately. Ratios above one mean the compact FlagQuantum path is
faster. All 41-sample measurements passed the configured stability check.

## What changed

A CX sequence is a linear map over GF(2). The compact representation stores the
inverse image of each basis bit instead of one source index per amplitude. At
22 qubits this changes the mapping metadata from `2^22` int32 entries
(16 MiB) to 22 int64 entries (176 bytes), a 95,325x reduction. Native forward
and adjoint kernels reconstruct source indices inside their existing state
traversals. Adjacent CNOT chains use a prefix-XOR specialization; arbitrary CX
networks use small eight-bit lookup chunks.

The isolated-process peak RSS fell from 699.312 MiB to 600.766 MiB on this run,
a measured reduction of 98.547 MiB (14.1%). Peak RSS includes imports, allocator
state, the circuit state, and one complete value-and-gradient call, so the
16 MiB table elimination is the directly attributable storage reduction.

Compact reconstruction costs additional integer operations. In this run it was
2.18% slower end to end than the full-index path. It is therefore an explicit
memory mode, not the default speed path. Set `FQ_NATIVE_CPU_COMPACT_CX_INDEX=1`
to enable it for CPU statevectors with at least 22 qubits. Smaller statevectors
continue to use the table path. Setting the variable to `0` is the rollback.

## What is measured and why it matters

The workload is a one-layer, 22-qubit hardware-efficient VQE circuit with 66
RX/RY/RZ parameters, a 21-gate nearest-neighbor CNOT chain, and 43 weighted Z/ZZ
Hamiltonian terms. The logical statevector is 64 MiB. Each retained sample
evaluates the expectation and obtains every gradient through Torch autograd and
FlagQuantum's exact reversible statevector-adjoint implementation.

This mode is useful when memory capacity matters more than the small timing
cost, including memory-constrained hosts and workloads whose full CX tables
would otherwise remain resident in the process cache. It is single-device
Apple arm64 CPU evidence, not a distributed scaling claim.

## FlagQuantum example

```python
import os

import torch

import flagquantum as fq
from flagquantum import algorithms as fqa

os.environ["FQ_NATIVE_CPU_COMPACT_CX_INDEX"] = "1"
theta = torch.full((22, 3), 0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(22, dtype=torch.complex128)
for wire in range(22):
    circuit.rx(wire, theta[wire, 0])
    circuit.ry(wire, theta[wire, 1])
    circuit.rz(wire, theta[wire, 2])
for wire in range(21):
    circuit.cx(wire, wire + 1)

hamiltonian = fqa.Hamiltonian(
    fqa.pauli_term(0.7, "ZZ", (wire, wire + 1)) for wire in range(21)
)
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduce

```bash
FQ_NATIVE_CPU_COMPACT_CX_INDEX=1 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 22 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
    flagquantum_adjoint_compact_cx_index_rollback \
    pennylane_lightning_adjoint \
  --warmup 5 --iterations 41 --calls-per-sample 1 \
  --json-output native_cpu_adjoint_compact_cx_cpu_arm64_20260930.json
```
