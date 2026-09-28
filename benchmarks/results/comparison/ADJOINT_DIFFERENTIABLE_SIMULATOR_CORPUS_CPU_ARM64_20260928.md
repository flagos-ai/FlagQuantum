# Adjoint differentiable simulator corpus (Apple arm64 CPU)

This report is generated from [`adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json`](adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json). It is a
separate method-matched track from the backpropagation corpus: FlagQuantum's
fused native reversible adjoint is compared with its Python fallback and
PennyLane Lightning's adjoint.
Both compute one exact weighted Z/ZZ expectation and its gradient with respect
to every circuit parameter; construction and optimizer updates are excluded.

## Results

| Workload | Qubits | Gates | Params | FQ forward (ms) | FQ backward (ms) | FQ total (ms) | Python backward (ms) | Python total (ms) | Native backward speedup | PennyLane Lightning total (ms) | PennyLane Lightning / FlagQuantum total | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 10 | 39 | 30 | 1.577 | 0.732 | 2.331 | 4.041 | 5.491 | 5.52x | 2.267 | 0.97x | 3.889e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 2.757 | 1.997 | 4.740 | 8.283 | 10.994 | 4.15x | 5.252 | 1.11x | 4.956e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 19.903 | 27.567 | 47.279 | 76.471 | 96.278 | 2.77x | 60.871 | 1.29x | 7.466e-14 |
| Hardware-efficient VQE | 22 | 87 | 66 | 461.624 | 524.822 | 965.937 | 1867.513 | 2307.161 | 3.56x | 1438.021 | 1.49x | 1.115e-12 |
| QAOA path MaxCut | 10 | 29 | 2 | 1.180 | 0.578 | 1.789 | 2.824 | 3.973 | 4.89x | 1.786 | 1.00x | 4.441e-16 |
| QAOA path MaxCut | 14 | 41 | 2 | 2.222 | 1.556 | 3.951 | 5.918 | 8.139 | 3.80x | 3.679 | 0.93x | 6.661e-16 |
| QAOA path MaxCut | 18 | 53 | 2 | 19.501 | 21.087 | 40.588 | 61.343 | 82.852 | 2.91x | 41.438 | 1.02x | 1.688e-14 |
| QAOA path MaxCut | 22 | 65 | 2 | 421.590 | 432.638 | 849.326 | 1336.359 | 1788.846 | 3.09x | 902.152 | 1.06x | 3.570e-13 |

A PennyLane Lightning/FlagQuantum ratio above one means FlagQuantum was faster. Peak RSS was
not measured in this timing run. These are local, non-release
single-device results, not a universal framework ranking or scaling claim.

## Meaning and current level

All 8 cases passed the 1e-9 value/gradient tolerance; maximum gradient error was 1.115e-12. One-pass forward RZZ segments and native disjoint one-qubit H/rotation layers complement the fused RX/RY/RZ/RZZ adjoint operators, commuting shared-parameter RZZ segments, shared rotation-layer adjoints with native shared-gradient reduction, allocation-free reverse H blocks, same-wire forward composition, CX-sequence permutations, reuse of the observable diagonal, bounded structural planning reuse, cached accelerator capability probes, and analytic Pauli-rotation VJPs. The native adjoint operators accelerated backward by 2.77x to 5.52x and total value-and-gradient by 2.04x to 2.39x. Total PennyLane Lightning/FlagQuantum ratios ranged from 0.93x to 1.49x on this host. Hardware-efficient VQE stresses many independent rotation gradients; QAOA stresses repeated shared parameters. This establishes functional local adjoint support for real weighted Z/ZZ Hamiltonians at complex128, not a universal performance or general Pauli-support claim. The QAOA path now reduces repeated planning and shared-parameter bookkeeping overhead while preserving the large-width execution path.

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
