# Adjoint differentiable simulator corpus (Apple arm64 CPU)

This report is generated from [`adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json`](adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json). It is a
separate method-matched track from the backpropagation corpus: FlagQuantum's
fused native reversible adjoint is compared with its Python fallback and
PennyLane Lightning's adjoint.
Both compute one exact weighted Z/ZZ expectation and its gradient with respect
to every circuit parameter; construction and optimizer updates are excluded.

## Results

| Workload | Qubits | Gates | Params | FQ value evaluation (ms) | FQ autograd callback (ms) | FQ value + gradient (ms) | Python autograd callback (ms) | Python value + gradient (ms) | Native callback speedup | PennyLane Lightning value + gradient (ms) | PennyLane Lightning / FlagQuantum value + gradient | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 10 | 39 | 30 | 1.271 | 0.668 | 1.951 | 3.987 | 5.241 | 5.97x | 2.539 | 1.30x | 3.889e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 2.567 | 2.129 | 4.921 | 8.676 | 11.383 | 4.08x | 5.936 | 1.21x | 4.956e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 20.474 | 28.159 | 48.421 | 93.834 | 115.117 | 3.33x | 65.486 | 1.35x | 7.466e-14 |
| Hardware-efficient VQE | 22 | 87 | 66 | 507.578 | 597.326 | 1104.905 | 2037.578 | 2527.226 | 3.41x | 1654.287 | 1.50x | 1.115e-12 |
| QAOA path MaxCut | 10 | 29 | 2 | 1.120 | 0.586 | 1.731 | 2.718 | 3.841 | 4.64x | 1.851 | 1.07x | 4.441e-16 |
| QAOA path MaxCut | 14 | 41 | 2 | 2.428 | 1.728 | 4.142 | 6.651 | 8.899 | 3.85x | 4.380 | 1.06x | 6.661e-16 |
| QAOA path MaxCut | 18 | 53 | 2 | 21.851 | 22.990 | 44.966 | 70.161 | 89.981 | 3.05x | 49.179 | 1.09x | 1.688e-14 |
| QAOA path MaxCut | 22 | 65 | 2 | 414.576 | 416.280 | 838.229 | 1294.972 | 1731.306 | 3.11x | 891.147 | 1.06x | 3.570e-13 |

Value evaluation and autograd callback are framework-observed Torch phases,
not method-matched adjoint-kernel boundaries. Lightning may perform most
adjoint derivative work during QNode value evaluation, leaving a very small
autograd callback. Only value + gradient is ranked across frameworks. A
PennyLane Lightning/FlagQuantum value + gradient ratio above one means
FlagQuantum was faster. Peak RSS was
not measured in this timing run. These are local, non-release
single-device results, not a universal framework ranking or scaling claim.

## Meaning and current level

All 8 cases passed the 1e-9 value/gradient tolerance; maximum gradient error was 1.115e-12. One-pass forward RZZ segments and native disjoint one-qubit H/rotation layers complement the fused RX/RY/RZ/RZZ adjoint operators, commuting shared-parameter RZZ segments, shared rotation-layer adjoints with native shared-gradient reduction, allocation-free reverse H blocks, same-wire forward composition, CX-sequence permutations, reuse of the observable diagonal, bounded structural planning reuse, cached accelerator capability probes, parameter-free adjoint IR template reuse, once-per-parameter binding validation, and analytic Pauli-rotation VJPs. The native adjoint operators accelerated backward by 3.05x to 5.97x and total value-and-gradient by 2.00x to 2.69x. Total PennyLane Lightning/FlagQuantum ratios ranged from 1.06x to 1.50x on this host. Hardware-efficient VQE stresses many independent rotation gradients; QAOA stresses repeated shared parameters. This establishes functional local adjoint support for real weighted Z/ZZ Hamiltonians at complex128, not a universal performance or general Pauli-support claim. The QAOA path now reduces repeated planning and shared-parameter bookkeeping overhead; repeated optimization steps also avoid rebuilding detached IR snapshots while preserving fresh parameter values and autograd contexts.

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
  --layers 1 --threads 1 --warmup 2 \
  --iterations 7 --calls-per-sample 1 \
  --skip-memory-probe \
  --json-output adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json --markdown-output REPORT.md
```
