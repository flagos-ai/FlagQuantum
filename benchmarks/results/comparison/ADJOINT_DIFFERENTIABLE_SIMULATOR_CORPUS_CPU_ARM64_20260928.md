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
| Hardware-efficient VQE | 10 | 39 | 30 | 1.824 | 0.743 | 2.551 | 3.938 | 5.998 | 5.30x | 2.160 | 0.85x | 3.889e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 3.368 | 2.169 | 5.474 | 8.518 | 11.550 | 3.93x | 5.469 | 1.00x | 4.956e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 20.864 | 28.471 | 49.637 | 79.072 | 100.040 | 2.78x | 63.646 | 1.28x | 7.466e-14 |
| Hardware-efficient VQE | 22 | 87 | 66 | 464.018 | 534.401 | 1006.355 | 1862.286 | 2283.085 | 3.48x | 1371.079 | 1.36x | 1.115e-12 |
| QAOA path MaxCut | 10 | 29 | 2 | 1.644 | 0.684 | 2.407 | 2.987 | 4.707 | 4.37x | 1.785 | 0.74x | 4.441e-16 |
| QAOA path MaxCut | 14 | 41 | 2 | 2.826 | 1.786 | 4.635 | 6.380 | 9.241 | 3.57x | 3.822 | 0.82x | 6.661e-16 |
| QAOA path MaxCut | 18 | 53 | 2 | 21.057 | 23.016 | 45.631 | 61.060 | 82.317 | 2.65x | 43.556 | 0.95x | 1.688e-14 |
| QAOA path MaxCut | 22 | 65 | 2 | 429.561 | 425.390 | 883.823 | 1320.905 | 1788.231 | 3.11x | 934.731 | 1.06x | 3.570e-13 |

A Lightning/FQ ratio above one means FlagQuantum was faster. Peak RSS was
not measured in this timing run. These are local, non-release
single-device results, not a universal framework ranking or scaling claim.

## Meaning and current level

All 8 cases passed the 1e-9 value/gradient tolerance; maximum gradient error was 1.115e-12. One-pass forward RZZ segments and native disjoint one-qubit H/rotation layers complement the fused RX/RY/RZ/RZZ adjoint operators, commuting shared-parameter RZZ segments, shared rotation-layer adjoints, allocation-free reverse H blocks, same-wire forward composition, CX-sequence permutations, reuse of the observable diagonal, and analytic Pauli-rotation VJPs. The native adjoint operators accelerated backward by 2.65x to 5.30x and total value-and-gradient by 1.80x to 2.35x. Total Lightning/FQ ratios ranged from 0.74x to 1.36x on this host. Hardware-efficient VQE stresses many independent rotation gradients; QAOA stresses repeated shared parameters. This establishes functional local adjoint support for real weighted Z/ZZ Hamiltonians at complex128, not a universal performance or general Pauli-support claim. The largest QAOA case now has balanced forward and backward costs; small-width framework overhead remains the next target.

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
