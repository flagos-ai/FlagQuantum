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
| Hardware-efficient VQE | 10 | 39 | 30 | 1.624 | 0.705 | 2.320 | 3.894 | 5.513 | 5.53x | 2.280 | 0.98x | 3.889e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 3.015 | 2.111 | 5.126 | 8.338 | 11.482 | 3.95x | 5.479 | 1.07x | 4.956e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 20.001 | 29.240 | 49.310 | 72.959 | 92.526 | 2.50x | 62.876 | 1.28x | 7.466e-14 |
| Hardware-efficient VQE | 22 | 87 | 66 | 416.961 | 523.955 | 940.916 | 1747.131 | 2204.782 | 3.33x | 1407.361 | 1.50x | 1.115e-12 |
| QAOA path MaxCut | 10 | 29 | 2 | 2.109 | 1.091 | 3.192 | 2.771 | 4.840 | 2.54x | 1.623 | 0.51x | 4.441e-16 |
| QAOA path MaxCut | 14 | 41 | 2 | 3.833 | 2.419 | 6.252 | 6.274 | 10.012 | 2.59x | 3.827 | 0.61x | 6.661e-16 |
| QAOA path MaxCut | 18 | 53 | 2 | 24.922 | 25.618 | 50.586 | 56.979 | 81.614 | 2.22x | 42.515 | 0.84x | 1.954e-14 |
| QAOA path MaxCut | 22 | 65 | 2 | 1157.658 | 877.870 | 2055.415 | 1626.083 | 2784.653 | 1.85x | 891.908 | 0.43x | 5.942e-13 |

A Lightning/FQ ratio above one means FlagQuantum was faster. Peak RSS was
not measured in this timing run. These are local, non-release
single-device results, not a universal framework ranking or scaling claim.

## Meaning and current level

All 8 cases passed the 1e-9 value/gradient tolerance; maximum gradient error was 1.115e-12. The fused RX/RY/RZ/RZZ CPU operators, commuting shared-parameter RZZ segments, same-wire forward composition, CX-sequence permutations, reuse of the observable diagonal, and analytic Pauli-rotation VJPs accelerated backward by 1.85x to 5.53x and total value-and-gradient by 1.35x to 2.38x. Total Lightning/FQ ratios ranged from 0.43x to 1.50x on this host. Hardware-efficient VQE stresses many independent rotation gradients; QAOA stresses repeated shared parameters. This establishes functional local adjoint support for real weighted Z/ZZ Hamiltonians at complex128, not performance parity or general Pauli support. QAOA remains forward-bound at the largest width; the next optimization target is native forward RZZ/RX layer fusion, followed by wider reverse rotation blocks.

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
