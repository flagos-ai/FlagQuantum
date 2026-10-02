# Native CPU adjoint adaptive parallel grain

This report is generated from
[`native_cpu_adjoint_parallel_grain_cpu_arm64_20261002.json`](native_cpu_adjoint_parallel_grain_cpu_arm64_20261002.json).
It measures one exact complex128 weighted Z/ZZ expectation and the complete
gradient of every circuit parameter. Each median retains 21 samples after three
warmups on Apple arm64 with eight CPU threads. Circuit construction and
optimizer updates are excluded.

## Results

| Workload | Qubits | Adaptive backward (ms) | 128-item rollback backward (ms) | Backward speedup | Adaptive value + gradient (ms) | Rollback value + gradient (ms) | Total speedup | PennyLane Lightning value + gradient (ms) | Lightning / FlagQuantum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 14 | 0.857 | 1.062 | 1.240x | 2.632 | 2.788 | 1.060x | 5.335 | 2.027x |
| Hardware-efficient VQE | 16 | 1.212 | 2.125 | 1.754x | 3.476 | 4.344 | 1.250x | 15.327 | 4.409x |
| Hardware-efficient VQE | 18 | 3.083 | 7.003 | 2.272x | 6.882 | 10.647 | 1.547x | 64.561 | 9.380x |
| Hardware-efficient VQE | 20 | 9.521 | 10.699 | 1.124x | 17.045 | 18.306 | 1.074x | 287.527 | 16.868x |
| Hardware-efficient VQE | 22 | 46.576 | 48.847 | 1.049x | 80.455 | 82.945 | 1.031x | 1448.716 | 18.007x |
| QAOA path MaxCut | 14 | 0.834 | 0.963 | 1.155x | 2.632 | 2.704 | 1.027x | 3.699 | 1.405x |
| QAOA path MaxCut | 16 | 1.566 | 2.083 | 1.330x | 3.646 | 4.127 | 1.132x | 10.308 | 2.827x |
| QAOA path MaxCut | 18 | 3.135 | 5.527 | 1.763x | 6.479 | 8.846 | 1.365x | 40.505 | 6.252x |
| QAOA path MaxCut | 20 | 12.946 | 14.587 | 1.127x | 21.686 | 23.811 | 1.098x | 197.433 | 9.104x |
| QAOA path MaxCut | 22 | 53.118 | 54.945 | 1.034x | 95.939 | 95.897 | 1.000x | 981.801 | 10.234x |

Ratios above one mean the optimized FlagQuantum path is faster. Adaptive
scheduling improves the complete value-and-gradient operation by 1.132x to
1.547x at 16--18 qubits, where the previous 128-item grain exposed too few
parallel tasks. The gain narrows at 20 qubits and is neutral at the 22-qubit
QAOA boundary because those wider states already expose enough tasks. This is
the intended guardrail: wide states retain the legacy grain instead of paying
extra scheduling overhead.

PennyLane Lightning may perform most adjoint derivative work during QNode value
evaluation, so its framework-observed callback is not compared with
FlagQuantum backward. The method-matched cross-framework metric is the complete
value plus gradient. FlagQuantum is 1.405x to 18.007x faster on all ten measured
cases. This is local, non-release evidence, not a universal framework ranking.

All cases passed the `1e-9` value/gradient tolerance and the stability contract;
the maximum observed Lightning gradient error was below `2e-12`.

## What changed

The fused native rotation adjoint previously used a fixed minimum grain of 128
tile blocks. At medium state widths that could leave fewer runnable tasks than
CPU workers. The kernel now targets about four tasks per worker while the tile
block count is below 1024, and keeps the established 128-item grain once the
state already supplies ample parallel work. Set
`FQ_NATIVE_CPU_ADJOINT_FINE_GRAIN=0` for the exact scheduling rollback.

## FlagQuantum example

No user-code change is required; the scheduler is internal to the native CPU
adjoint path.

```python
import torch

import flagquantum as fq
from flagquantum import algorithms as fqa

torch.set_num_threads(8)
theta = torch.full((18, 3), 0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(18, dtype=torch.complex128)
for wire in range(18):
    circuit.rx(wire, theta[wire, 0])
    circuit.ry(wire, theta[wire, 1])
    circuit.rz(wire, theta[wire, 2])
for wire in range(17):
    circuit.cx(wire, wire + 1)
hamiltonian = fqa.Hamiltonian(
    tuple(fqa.pauli_term(0.7, "ZZ", (wire, wire + 1)) for wire in range(17))
    + tuple(fqa.pauli_term(0.2, "Z", (wire,)) for wire in range(18))
)
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduce

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 14 16 18 20 22 --layers 1 --threads 8 \
  --engines flagquantum_adjoint \
    flagquantum_adjoint_parallel_grain_rollback \
    pennylane_lightning_adjoint \
  --warmup 3 --iterations 21 --calls-per-sample 1 \
  --skip-memory-probe \
  --json-output native_cpu_adjoint_parallel_grain.json
```
