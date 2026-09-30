# Native CPU adjoint terminal-layer comparison

This report is generated from
[`native_cpu_adjoint_terminal_no_restore_cpu_arm64_20260929.json`](native_cpu_adjoint_terminal_no_restore_cpu_arm64_20260929.json).
It compares the optimized native CPU adjoint with an exact runtime rollback
that restores ket and adjoint after the earliest parameterized Euler layer.
Both paths execute the same FlagQuantum circuit, observable, complex128
statevector-adjoint method, thread count, and full parameter gradient.

| Workload | Qubits | Parameters | CPU threads | Optimized backward (ms) | Restore rollback backward (ms) | Backward speedup | Optimized value + grad (ms) | Restore rollback value + grad (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 22 | 66 | 2 | 98.069 | 137.573 | 1.403x | 158.140 | 197.563 | 1.249x | 4.947e-15 |

## What changed

The native reverse sweep recognizes a complete earliest `RX-RY-RZ` layer.
Once all gradients in that layer have been accumulated, no earlier
parameterized operation can consume the reconstructed ket or adjoint. The
kernel therefore omits the combined inverse updates and final state writes.

When a nearest-neighbor CNOT layer immediately follows the rotations, its
inverse permutation is consumed while the first rotation tile is loaded. This
avoids materializing a separate intermediate ket and adjoint. The Euler path
also evaluates the X, Y, and Z Pauli bilinears once per amplitude pair and
reuses them for all three gradients.

Set `FQ_NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE=0` to restore the exact previous
behavior. Set `FQ_NATIVE_CPU_ADJOINT_CX_ROTATION_FUSION=0` to materialize the
CNOT inverse separately.

## What is measured and why it matters

Each retained sample evaluates the exact expectation value and computes the
complete 66-parameter adjoint gradient of a one-layer, 22-qubit
hardware-efficient VQE circuit. The circuit contains 66 `RX/RY/RZ` rotations
and a 21-gate nearest-neighbor CNOT chain; the Hamiltonian contains 43 weighted
Z/ZZ terms. The logical complex128 statevector is 64 MiB.

The medians use 5 warmups and 41 retained samples in one process with rotated
engine order. Values and full gradients are checked on every path. Backward
improves by 40.28%, while end-to-end value plus gradient improves by 24.93%.
Forward time is unchanged within noise (59.625 vs 59.103 ms).

## Correctness boundary

The no-restore path is enabled only when the reverse sweep reaches the start of
the circuit and the entire segment consists of complete, contiguous
`RZ-RY-RX` reverse triples. Other layouts retain full state restoration. The
kernel rejects an invalid no-restore request, and both complex64 and complex128
are covered by direct fused-versus-sequential tests.

This is measured single-device Apple arm64 CPU evidence with two PyTorch
threads. It is not a distributed scaling result or a universal circuit
speedup. Deeper circuits benefit at their earliest compatible Euler layer;
their other layers still require reversible state reconstruction.

## FlagQuantum example

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

parameters = torch.full(
    (6, 3), 0.2, dtype=torch.float64, requires_grad=True
)
circuit = fq.Circuit(6, dtype=torch.complex128)
for qubit in range(6):
    circuit.rx(qubit, parameters[qubit, 0])
    circuit.ry(qubit, parameters[qubit, 1])
    circuit.rz(qubit, parameters[qubit, 2])
for qubit in range(5):
    circuit.cx(qubit, qubit + 1)

hamiltonian = fqa.Hamiltonian(
    fqa.HamiltonianTerm(0.7, {qubit: "z", qubit + 1: "z"})
    for qubit in range(5)
)
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), parameters.grad)
```

No API change is required; compatible native CPU adjoint circuits use the
optimized terminal path automatically.

## Reproduce

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe \
  --n-wires 22 \
  --engines flagquantum_adjoint flagquantum_adjoint_terminal_no_restore_rollback \
  --layers 1 --threads 2 \
  --warmup 5 --iterations 41 \
  --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_adjoint_terminal_no_restore_cpu_arm64_20260929.json
```
