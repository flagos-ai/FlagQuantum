# Differentiable simulator workload corpus (Apple arm64 CPU)

This report is generated from [`differentiable_simulator_corpus_cpu_arm64_20260928.json`](differentiable_simulator_corpus_cpu_arm64_20260928.json).
It measures one exact expectation-value forward pass and the full gradient
with respect to every circuit parameter. Circuit/device construction and an
optimizer update are excluded. Each engine uses its native Torch-compatible
gradient path: PyTorch reverse-mode statevector backpropagation for both
FlagQuantum and PennyLane default.qubit.

Each median uses 9 retained samples after 1 warmup run, with 1 call per sample.
Environment: macOS-27.0-arm64-arm-64bit; Python 3.12.14; flagquantum 0.2.0, torch
2.13.0, pennylane 0.45.1.

## Results

| Workload | Qubits | Gates | Params | FQ forward (ms) | FQ backward (ms) | FQ total (ms) | PL forward (ms) | PL backward (ms) | PL total (ms) | PL / FQ total | FQ peak RSS (MiB) | PL peak RSS (MiB) | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 10 | 39 | 30 | 1.766 | 1.503 | 3.294 | 3.371 | 2.046 | 5.505 | 1.67x | 203.8 | 412.7 | 3.023e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 4.091 | 5.343 | 9.362 | 7.024 | 7.809 | 14.921 | 1.59x | 213.5 | 422.6 | 3.028e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 40.763 | 65.308 | 105.909 | 46.085 | 108.640 | 155.339 | 1.47x | 378.7 | 655.9 | 7.179e-15 |
| Hardware-efficient VQE | 22 | 87 | 66 | 1132.953 | 1769.963 | 2947.274 | 1421.368 | 3403.051 | 4989.496 | 1.69x | 3415.8 | 3713.7 | 4.544e-15 |
| QAOA path MaxCut | 10 | 29 | 2 | 1.378 | 1.559 | 3.004 | 2.674 | 1.365 | 4.095 | 1.36x | 203.5 | 411.7 | 1.332e-15 |
| QAOA path MaxCut | 14 | 41 | 2 | 3.348 | 3.845 | 7.271 | 6.248 | 5.489 | 11.600 | 1.60x | 209.6 | 419.7 | 5.551e-16 |
| QAOA path MaxCut | 18 | 53 | 2 | 35.181 | 45.763 | 80.960 | 54.889 | 86.339 | 141.573 | 1.75x | 336.9 | 604.0 | 1.048e-13 |
| QAOA path MaxCut | 22 | 65 | 2 | 1025.693 | 1723.961 | 3008.643 | 6920.977 | 4265.360 | 11396.678 | 3.79x | 2473.3 | 2947.8 | 1.277e-12 |

A PL/FQ ratio above one means FlagQuantum was faster; below one means
PennyLane default.qubit was faster. Peak RSS is measured in a separate isolated
process and includes framework import, circuit/device construction, one warmup,
and one value-and-gradient execution, so it is an operational footprint rather
than tensor-only memory. These local results are not a universal framework
ranking, optimizer-throughput claim, or scalability evidence.

## Interpretation

All 8 measured cases passed the value-and-gradient check and the 20% RMAD stability
gate; the maximum gradient error was 1.277e-12. FlagQuantum completed the matched value-
and-gradient operation 1.36x to 3.79x faster on this host.

At 22 qubits, FlagQuantum's forward and total execution were faster than default.qubit,
while native backward remained its largest component (Hardware-efficient VQE: forward
1132.953 ms and backward 1769.963 ms; QAOA path MaxCut: forward 1025.693 ms and backward
1723.961 ms). The measured optimization target is therefore reverse-mode state retention
and backward execution, not another forward-only gate kernel. FlagQuantum's isolated
peak RSS reached 2.42-3.34 GiB at this width, versus 2.88-3.63 GiB for PennyLane
default.qubit.

## What the workloads mean

- **Hardware-efficient VQE** applies one RX-RY-RZ layer and a linear CNOT
  chain, then differentiates a nearest-neighbour Ising energy. It represents
  the inner quantum step of variational energy minimization.
- **QAOA path MaxCut** prepares one exact QAOA layer for a path graph and
  differentiates its expected cut value. Shared beta and gamma parameters
  separate state-size cost from parameter-count growth.

Qiskit Aer and Cirq are intentionally absent: the current FlagQuantum bridges
do not expose a native PyTorch gradient contract. Substituting finite differences
would measure a different algorithm and would not be a fair comparison.

## FlagQuantum example

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

theta = torch.full((4, 3), 0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(4, dtype=torch.complex128)
for wire in range(4):
    circuit.rx(wire, theta[wire, 0])
    circuit.ry(wire, theta[wire, 1])
    circuit.rz(wire, theta[wire, 2])
for wire in range(3):
    circuit.cx(wire, wire + 1)

hamiltonian = fqa.Hamiltonian(
    fqa.HamiltonianTerm(0.7, {wire: "z", wire + 1: "z"})
    for wire in range(3)
)
energy = hamiltonian.expectation(circuit.state(refresh=True)).sum()
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 10 14 18 22 \
  --layers 1 --threads 1 \
  --warmup 1 --iterations 9 \
  --calls-per-sample 1 \
  --json-output differentiable_simulator_corpus_cpu_arm64_20260928.json --markdown-output REPORT.md
```
