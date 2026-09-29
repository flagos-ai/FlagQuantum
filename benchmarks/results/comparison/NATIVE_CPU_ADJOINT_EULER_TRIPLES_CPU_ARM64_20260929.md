# Native CPU adjoint Euler-triple comparison

This report is generated from [`native_cpu_adjoint_euler_triples_cpu_arm64_20260929.json`](native_cpu_adjoint_euler_triples_cpu_arm64_20260929.json). It compares
FlagQuantum's native reverse sweep with and without the same-wire RZ/RY/RX
Euler-triple and branchless amplitude-pair kernel. Both paths use exact
statevector adjoint differentiation and must return matching values and full
parameter gradients. Kernel tests also compare ket and adjoint states.
Ratios above 1 mean the fused path is faster.

| Workload | Qubits | Euler triples | Native backward (ms) | Rollback backward (ms) | Backward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 22 | 22 | 54.876 | 70.876 | 1.29x | 87.397 | 104.122 | 1.19x | 5.551e-17 |
| QAOA path MaxCut | 22 | 0 | 84.282 | 87.377 | 1.04x | 147.343 | 150.455 | 1.02x | 8.882e-16 |

## What this measures

Hardware-efficient VQE applies RX, RY, and RZ consecutively to each
qubit, which the reverse sweep visits as RZ/RY/RX. The fused kernel keeps
one amplitude pair in registers while computing three gradients and undoing
all three rotations, instead of loading and storing the pair once per gate.
The pair loop uses an OpenMP SIMD reduction for its gradient totals. All other
rotations use direct zero/one pair blocks without a per-amplitude bit test.
QAOA has no Euler triples, so it isolates that branchless traversal gain.
This is non-release single-device evidence, not a thread-scaling or
cross-machine claim.

## FlagQuantum example

The optimization is automatic behind the existing adjoint API; user code
does not select a kernel:

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(3, dtype=torch.complex128)
for wire in range(3):
    circuit.rx(wire, theta).ry(wire, theta).rz(wire, theta)
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
  --engines flagquantum_adjoint flagquantum_adjoint_euler_triple_rollback \
  --layers 1 --threads 8 \
  --warmup 3 --iterations 11 \
  --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_adjoint_euler_triples_cpu_arm64_20260929.json --markdown-output REPORT.md
```

Set `FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES=0` to restore the prior
per-gate, bit-tested pair traversal while retaining native adjoint.
