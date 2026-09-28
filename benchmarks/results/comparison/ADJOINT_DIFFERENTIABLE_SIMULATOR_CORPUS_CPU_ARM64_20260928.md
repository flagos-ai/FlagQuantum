# Adjoint differentiable simulator corpus (Apple arm64 CPU)

This report is generated from [`adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json`](adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json). It is a
separate method-matched track from the backpropagation corpus: FlagQuantum's
fused native reversible adjoint is compared with its Python fallback and
PennyLane Lightning's adjoint.
Both compute one exact weighted Z/ZZ expectation and its gradient with respect
to every circuit parameter; construction and optimizer updates are excluded.

## Results

| Workload | Qubits | Gates | Params | FQ forward (ms) | FQ backward (ms) | FQ total (ms) | Python backward (ms) | Python total (ms) | Native backward speedup | Lightning total (ms) | Lightning / FQ total | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 10 | 39 | 30 | 1.806 | 1.781 | 3.535 | 4.282 | 6.215 | 2.40x | 2.311 | 0.65x | 3.727e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 3.297 | 3.500 | 6.702 | 8.893 | 11.965 | 2.54x | 5.422 | 0.81x | 5.310e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 22.067 | 30.835 | 52.988 | 93.948 | 116.107 | 3.05x | 64.468 | 1.22x | 7.466e-14 |
| Hardware-efficient VQE | 22 | 87 | 66 | 457.867 | 560.152 | 1026.943 | 1908.816 | 2375.371 | 3.41x | 1483.966 | 1.45x | 1.115e-12 |
| QAOA path MaxCut | 10 | 29 | 2 | 2.273 | 1.220 | 3.468 | 2.789 | 5.195 | 2.29x | 1.716 | 0.49x | 4.441e-16 |
| QAOA path MaxCut | 14 | 41 | 2 | 3.885 | 2.444 | 6.390 | 6.569 | 10.455 | 2.69x | 3.888 | 0.61x | 6.661e-16 |
| QAOA path MaxCut | 18 | 53 | 2 | 29.340 | 27.630 | 56.057 | 67.828 | 96.979 | 2.45x | 42.315 | 0.75x | 1.954e-14 |
| QAOA path MaxCut | 22 | 65 | 2 | 1174.694 | 869.070 | 2041.893 | 1686.212 | 2877.651 | 1.94x | 938.091 | 0.46x | 5.942e-13 |

A Lightning/FQ ratio above one means FlagQuantum was faster. Peak RSS was
not measured in this timing run. These are local, non-release
single-device results, not a universal framework ranking or scaling claim.

## Meaning and current level

All 8 cases passed the 1e-9 value/gradient tolerance; maximum gradient error was 1.115e-12. The fused RX/RY/RZ/RZZ CPU operators, commuting shared-parameter RZZ segments, same-wire forward composition, CX-sequence permutations, reuse of the observable diagonal, and analytic Pauli-rotation VJPs accelerated backward by 1.94x to 3.41x and total value-and-gradient by 1.41x to 2.31x. Total Lightning/FQ ratios ranged from 0.46x to 1.45x on this host. Hardware-efficient VQE stresses many independent rotation gradients; QAOA stresses repeated shared parameters. This establishes functional local adjoint support for real weighted Z/ZZ Hamiltonians at complex128, not performance parity or general Pauli support. QAOA remains forward-bound at the largest width; the next optimization target is native forward RZZ/RX layer fusion, followed by wider reverse rotation blocks.

## FlagQuantum example

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(3, dtype=torch.complex128).ry(0, theta).cx(0, 1)
hamiltonian = fqa.Hamiltonian((
    fqa.pauli_term(0.7, "ZZ", (0, 1)),
    fqa.pauli_term(0.2, "Z", (2,)),
))
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --engines flagquantum_adjoint flagquantum_adjoint_python_fallback pennylane_lightning_adjoint \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 10 14 18 22 \
  --layers 1 --threads 1 --warmup 1 \
  --iterations 5 --calls-per-sample 1 \
  --skip-memory-probe \
  --json-output adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json --markdown-output REPORT.md
```
