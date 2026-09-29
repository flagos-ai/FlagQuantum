# Native CPU shared-RZZ comparison

This report is generated from
[`native_cpu_shared_rzz_cpu_arm64_20260929.json`](native_cpu_shared_rzz_cpu_arm64_20260929.json).
It measures the exact 22-qubit QAOA path-MaxCut workload before and after three
related native CPU changes: exact-view parameter deduplication, a lookup table
for repeated shared-RZZ phases, and fusion of the terminal RX reverse tile with
the adjacent shared-RZZ reverse step. Ratios above one mean the optimized path
is faster.

| Workload | Qubits | Gates | Parameters | Optimized forward (ms) | Rollback forward (ms) | Forward speedup | Optimized backward (ms) | Rollback backward (ms) | Backward speedup | Optimized total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QAOA path MaxCut | 22 | 65 | 2 | 83.328 | 91.798 | 1.10x | 97.887 | 102.182 | 1.04x | 180.598 | 196.709 | 1.09x | 1.740e-12 |

Both paths passed the `1e-9` correctness threshold. Nine retained samples
followed two warmups; total-time relative median absolute deviation was 14.72%
for the optimized path and 5.19% for rollback, both below the benchmark's 20%
stability ceiling. This is measured comparison evidence on one Apple arm64 CPU,
not a cross-machine scalability claim.

## What this improves

The QAOA circuit reuses one scalar parameter for every RZZ edge. Tensor indexing
creates distinct Python view objects for that same storage location, so the old
layout treated 21 uses as unrelated parameters. The new layout recognizes exact
aliases by device, dtype, data pointer, shape, and stride while keeping genuinely
independent equal-valued tensors separate.

With that shared-parameter contract restored, the forward kernel precomputes the
22 possible complex phases instead of evaluating `sin` and `cos` for every one
of the 4,194,304 amplitudes. In backward, the last RX tile also applies the
adjacent RZZ inverse and accumulates its shared gradient before writing the state,
removing a separate full-state traversal. Unsupported layouts and non-shared RZZ
segments retain the existing kernels.

## FlagQuantum example

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

parameters = torch.tensor([0.31, 0.17], dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(6, dtype=torch.complex128)
for wire in range(6):
    circuit.h(wire)
for wire in range(5):
    circuit.rzz(wire, wire + 1, parameters[0])
for wire in range(6):
    circuit.rx(wire, parameters[1])

hamiltonian = fqa.Hamiltonian(
    fqa.HamiltonianTerm(-0.5, {wire: "z", wire + 1: "z"})
    for wire in range(5)
)
energy = hamiltonian.expectation(circuit, differentiation="adjoint") + 2.5
energy.backward()
print(energy.item(), parameters.grad)
```

## Reproduce

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads qaoa_path_maxcut --n-wires 22 --layers 1 --threads 4 \
  --engines flagquantum_adjoint flagquantum_adjoint_shared_rzz_rollback \
  --warmup 2 --iterations 9 --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_shared_rzz_cpu_arm64_20260929.json
```

Set `FQ_NATIVE_CPU_ADJOINT_RX_RZZ_FUSION=0` for direct rollback of the shared
phase lookup and RX-to-RZZ reverse fusion. Exact-view parameter deduplication is
a semantic correction and remains active in both paths.
