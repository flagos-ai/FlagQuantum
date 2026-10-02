# Native CPU adjoint saved-parameter validation

This report is generated from
[`native_cpu_adjoint_saved_parameter_validation_cpu_arm64_20261002.json`](native_cpu_adjoint_saved_parameter_validation_cpu_arm64_20261002.json).
It measures one exact complex128 weighted Z/ZZ expectation and the complete
gradient of every circuit parameter. Each median retains 21 samples after three
warmups on Apple arm64 with one CPU thread. Circuit construction and optimizer
updates are excluded.

## Results

| Workload | Qubits | Optimized backward (ms) | Revalidation rollback backward (ms) | Backward speedup | Optimized value + gradient (ms) | Rollback value + gradient (ms) | Total speedup | PennyLane Lightning value + gradient (ms) | Lightning / FlagQuantum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 10 | 0.579 | 0.701 | 1.209x | 1.961 | 1.951 | 0.995x | 2.434 | 1.241x |
| Hardware-efficient VQE | 14 | 1.165 | 1.318 | 1.132x | 2.983 | 3.608 | 1.209x | 6.318 | 2.118x |
| Hardware-efficient VQE | 18 | 10.476 | 10.657 | 1.017x | 19.122 | 19.256 | 1.007x | 72.444 | 3.789x |
| Hardware-efficient VQE | 22 | 199.498 | 198.364 | 0.994x | 327.984 | 324.259 | 0.989x | 1565.935 | 4.774x |
| QAOA path MaxCut | 10 | 0.302 | 0.308 | 1.021x | 1.531 | 1.464 | 0.957x | 1.823 | 1.191x |
| QAOA path MaxCut | 14 | 1.019 | 1.060 | 1.040x | 2.700 | 2.935 | 1.087x | 4.327 | 1.603x |
| QAOA path MaxCut | 18 | 12.174 | 11.898 | 0.977x | 21.887 | 21.929 | 1.002x | 44.283 | 2.023x |
| QAOA path MaxCut | 22 | 210.027 | 213.625 | 1.017x | 367.603 | 366.038 | 0.996x | 949.361 | 2.583x |

Ratios above one mean the optimized FlagQuantum path is faster. The change
removes a fixed-cost validation pass, so its clearest effect is the 1.209x and
1.132x hardware-efficient backward speedup at 10 and 14 qubits. At 18 and 22
qubits the native fused adjoint kernel dominates, and the observed difference
is within roughly 2.3% in either direction; this report does not claim a
wide-state kernel speedup.

PennyLane Lightning may perform most adjoint derivative work during QNode value
evaluation, leaving a much smaller framework-observed autograd callback. Its
callback time is therefore not compared with FlagQuantum backward in this
table. The method-matched cross-framework metric is the complete value plus
gradient operation. FlagQuantum is 1.191x to 4.774x faster on all eight measured
cases. This is local, non-release evidence, not a universal framework ranking.

All cases passed the `1e-9` value/gradient tolerance and the stability contract;
the maximum observed Lightning gradient error was `1.780e-12`.

## What changed

The public adjoint forward pass already validates every trainable gate
parameter before saving it in the custom-autograd context. Backward receives
those exact saved tensors, and PyTorch's saved-tensor version check rejects an
in-place mutation before the callback runs. Repeating finite/range validation
while rebinding the detached backward IR was therefore redundant. Backward now
reuses the forward validation result. Set
`FQ_STATEVECTOR_ADJOINT_REVALIDATE_SAVED_PARAMETERS=1` for the exact rollback.
Every new execution still validates current values, including a value changed
between optimization steps.

## FlagQuantum example

```python
import torch

import flagquantum as fq
from flagquantum import algorithms as fqa

theta = torch.full((3, 3), 0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(3, dtype=torch.complex128)
for wire in range(3):
    circuit.rx(wire, theta[wire, 0])
    circuit.ry(wire, theta[wire, 1])
    circuit.rz(wire, theta[wire, 2])
circuit.cx(0, 1).cx(1, 2)
hamiltonian = fqa.Hamiltonian((
    fqa.pauli_term(0.7, "ZZ", (0, 1)),
    fqa.pauli_term(0.2, "Z", (2,)),
))
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 10 14 18 22 --layers 1 --threads 1 \
  --engines flagquantum_adjoint \
    flagquantum_adjoint_saved_parameter_revalidation \
    pennylane_lightning_adjoint \
  --warmup 3 --iterations 21 --calls-per-sample 1 \
  --skip-memory-probe \
  --json-output native_cpu_adjoint_saved_parameter_validation.json
```
