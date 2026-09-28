# Adjoint differentiable simulator corpus (Apple arm64 CPU)

This report is generated from [`adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json`](adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json). It is a
separate method-matched track from the backpropagation corpus: FlagQuantum's
reversible statevector adjoint is compared with PennyLane Lightning's adjoint.
Both compute one exact weighted Z/ZZ expectation and its gradient with respect
to every circuit parameter; construction and optimizer updates are excluded.

## Results

| Workload | Qubits | Gates | Params | FQ forward (ms) | FQ backward (ms) | FQ total (ms) | Rollback backward (ms) | Rollback total (ms) | Backward speedup | Lightning total (ms) | Lightning / FQ total | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 10 | 39 | 30 | 2.199 | 3.984 | 6.159 | 7.302 | 9.982 | 1.83x | 2.340 | 0.38x | 1.906e-15 |
| Hardware-efficient VQE | 14 | 55 | 42 | 3.878 | 8.450 | 12.329 | 37.812 | 46.350 | 4.47x | 5.604 | 0.45x | 2.998e-15 |
| Hardware-efficient VQE | 18 | 71 | 54 | 26.575 | 79.755 | 106.330 | 576.072 | 684.147 | 7.22x | 64.546 | 0.61x | 1.117e-14 |
| Hardware-efficient VQE | 22 | 87 | 66 | 572.547 | 1789.205 | 2361.753 | 18782.621 | 22801.422 | 10.50x | 1401.022 | 0.59x | 9.536e-13 |
| QAOA path MaxCut | 10 | 29 | 2 | 2.402 | 2.658 | 5.249 | 4.923 | 7.492 | 1.85x | 1.794 | 0.34x | 4.441e-16 |
| QAOA path MaxCut | 14 | 41 | 2 | 3.706 | 5.824 | 9.685 | 27.273 | 32.133 | 4.68x | 3.877 | 0.40x | 1.443e-15 |
| QAOA path MaxCut | 18 | 53 | 2 | 24.994 | 60.560 | 85.662 | 422.897 | 466.137 | 6.98x | 42.274 | 0.49x | 2.798e-14 |
| QAOA path MaxCut | 22 | 65 | 2 | 1159.628 | 1710.723 | 2874.856 | 13589.266 | 14908.373 | 7.94x | 908.103 | 0.32x | 1.759e-12 |

A Lightning/FQ ratio above one means FlagQuantum was faster. Peak RSS was
not measured in this timing run. These are local, non-release
single-device results, not a universal framework ranking or scaling claim.

## Meaning and current level

All 8 cases passed the 1e-9 value/gradient tolerance; maximum gradient error was 1.759e-12. Native CPU layouts, same-wire forward composition, CX-sequence permutations, reuse of the observable diagonal, and analytic Pauli-rotation VJPs accelerated backward by 1.83x to 10.50x and total value-and-gradient by 1.43x to 9.65x. Total Lightning/FQ ratios ranged from 0.32x to 0.61x on this host, so Lightning was 1.65x to 3.17x faster. Hardware-efficient VQE stresses many independent rotation gradients; QAOA stresses repeated shared parameters. This establishes functional local adjoint support for real weighted Z/ZZ Hamiltonians at complex128, not performance parity or general Pauli support. The next optimization target is block-fusing commuting rotation regions and moving the remaining hot reverse sweep into compiled native kernels.

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
  --engines flagquantum_adjoint flagquantum_adjoint_gather_rollback pennylane_lightning_adjoint \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 10 14 18 22 \
  --layers 1 --threads 1 --warmup 1 \
  --iterations 5 --calls-per-sample 1 \
  --skip-memory-probe \
  --json-output adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json --markdown-output REPORT.md
```
