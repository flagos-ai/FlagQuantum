# Native CPU adjoint observable-boundary comparison

This report is generated from [`native_cpu_adjoint_observable_boundary_cpu_arm64_20260929.json`](native_cpu_adjoint_observable_boundary_cpu_arm64_20260929.json). It compares
FlagQuantum's fused native observable expectation and adjoint-seed kernels
against the prior eager PyTorch tensor expressions. Circuit execution, cached
Z/ZZ observable weights, and the remaining adjoint sweep are identical.
Ratios above 1 mean the fused path is faster.

| Workload | Qubits | Observable terms | Native forward (ms) | Rollback forward (ms) | Forward speedup | Native backward (ms) | Rollback backward (ms) | Backward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max value error | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 22 | 43 | 36.781 | 41.214 | 1.12x | 73.917 | 79.113 | 1.07x | 110.992 | 120.209 | 1.08x | 1.815e-11 | 0.000e+00 |
| QAOA path MaxCut | 22 | 22 | 75.234 | 75.979 | 1.01x | 112.435 | 116.401 | 1.04x | 189.000 | 192.695 | 1.02x | 1.599e-14 | 0.000e+00 |

## What this measures

Exact Z/ZZ expectations multiply every state probability by a cached real
diagonal. Reverse-mode differentiation then seeds the adjoint with twice the
state times that same diagonal. The native boundary performs each operation
in one parallel traversal, avoiding eager full-state intermediates. It is
automatic for contiguous complex64/complex128 CPU statevectors; unsupported
layouts retain the existing PyTorch fallback. These are local single-device
measurements, not distributed or cross-machine scalability claims.

## FlagQuantum example

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(3, dtype=torch.complex128)
for wire in range(3):
    circuit.rx(wire, theta).ry(wire, theta).rz(wire, theta)
circuit.cx(0, 1).cx(1, 2)
hamiltonian = fqa.Hamiltonian((fqa.pauli_term(1.0, "ZZ", (0, 1)),))
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduce

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 22 \
  --engines flagquantum_adjoint flagquantum_adjoint_observable_boundary_rollback \
  --layers 1 --threads 4 \
  --warmup 2 --iterations 9 \
  --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_adjoint_observable_boundary_cpu_arm64_20260929.json --markdown-output REPORT.md
```

Set `FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY=0` to restore the eager expectation
and adjoint-seed expressions while retaining the rest of native adjoint.
