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
| Hardware-efficient VQE | 10 | 39 | 30 | 2.163 | 1.598 | 3.777 | 3.960 | 6.084 | 2.48x | 2.289 | 0.61x | 3.986e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 3.791 | 3.418 | 7.211 | 8.434 | 12.204 | 2.47x | 5.415 | 0.75x | 4.181e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 26.232 | 29.374 | 55.596 | 79.431 | 105.714 | 2.70x | 60.601 | 1.09x | 7.455e-14 |
| Hardware-efficient VQE | 22 | 87 | 66 | 585.909 | 521.844 | 1131.688 | 1760.283 | 2343.998 | 3.37x | 1362.749 | 1.20x | 1.115e-12 |
| QAOA path MaxCut | 10 | 29 | 2 | 2.136 | 1.045 | 3.264 | 2.684 | 4.777 | 2.57x | 1.645 | 0.50x | 4.441e-16 |
| QAOA path MaxCut | 14 | 41 | 2 | 3.788 | 2.419 | 6.243 | 6.073 | 9.770 | 2.51x | 3.709 | 0.59x | 6.661e-16 |
| QAOA path MaxCut | 18 | 53 | 2 | 26.359 | 26.310 | 52.999 | 59.071 | 85.055 | 2.25x | 40.280 | 0.76x | 1.954e-14 |
| QAOA path MaxCut | 22 | 65 | 2 | 1189.180 | 902.644 | 2094.194 | 1732.681 | 2897.119 | 1.92x | 901.280 | 0.43x | 5.942e-13 |

A Lightning/FQ ratio above one means FlagQuantum was faster. Peak RSS was
not measured in this timing run. These are local, non-release
single-device results, not a universal framework ranking or scaling claim.

## Meaning and current level

All 8 cases passed the 1e-9 value/gradient tolerance; maximum gradient error was 1.115e-12. The fused RX/RY/RZ/RZZ CPU operators, commuting shared-parameter RZZ segments, same-wire forward composition, CX-sequence permutations, reuse of the observable diagonal, and analytic Pauli-rotation VJPs accelerated backward by 1.92x to 3.37x and total value-and-gradient by 1.38x to 2.07x. Total Lightning/FQ ratios ranged from 0.43x to 1.20x on this host. Hardware-efficient VQE stresses many independent rotation gradients; QAOA stresses repeated shared parameters. This establishes functional local adjoint support for real weighted Z/ZZ Hamiltonians at complex128, not performance parity or general Pauli support. QAOA remains forward-bound at the largest width; the next optimization target is native forward RZZ/RX layer fusion, followed by wider reverse rotation blocks.

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
