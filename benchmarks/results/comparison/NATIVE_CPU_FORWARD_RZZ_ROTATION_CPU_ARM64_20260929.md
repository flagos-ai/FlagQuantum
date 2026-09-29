# Native CPU QAOA boundary comparison

This report is generated from
[`native_cpu_forward_rzz_rotation_cpu_arm64_20260929.json`](native_cpu_forward_rzz_rotation_cpu_arm64_20260929.json).
It measures an exact 22-qubit QAOA path-MaxCut value-and-gradient evaluation
before and after the native CPU forward and backward boundary changes in this
PR. Ratios above one mean the optimized path is faster.

| Workload | Qubits | Gates | Parameters | Optimized forward (ms) | Rollback forward (ms) | Forward speedup | Optimized backward (ms) | Rollback backward (ms) | Backward speedup | Optimized total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QAOA path MaxCut | 22 | 65 | 2 | 66.208 | 119.486 | 1.80x | 65.829 | 95.549 | 1.45x | 134.134 | 213.654 | 1.59x | 1.776e-15 |

Both paths passed the `1e-9` correctness threshold. Thirty-one retained samples
followed five warmups. Total-time relative median absolute deviation was 3.71%
for the optimized path and 1.24% for rollback, both below the benchmark's 20%
stability ceiling.

## What is measured and why it matters

The workload starts every qubit in superposition, applies a nearest-neighbor
shared-parameter RZZ cost layer, applies one RX mixer per qubit, evaluates the
path-MaxCut Hamiltonian, and computes the full two-parameter adjoint gradient.
It therefore exercises a common QAOA boundary rather than an isolated synthetic
gate microbenchmark.

The optimized forward path fuses the shared RZZ layer into the first following
rotation tile and replaces generic complex matrix multiplication with
structurally equivalent RX, RY, and RZ arithmetic. Contiguous butterfly pairs
also remove the legacy per-amplitude bit-test branch. In reverse, the last
rotation tile now absorbs the preceding Hadamards after undoing its adjacent
RZZ layer. That reduces the ket-plus-adjoint full-state scans for this QAOA
boundary from four to three. Together, these changes reduce memory traffic and
arithmetic without changing the public API or the fallback path.

## Current level

On this Apple arm64 CPU with four PyTorch threads, the optimized FlagQuantum
forward pass takes 44.6% less time and backward takes 31.1% less time than the
direct rollback. The complete value-and-gradient call takes 37.2% less time.
This is useful evidence for the measured 22-qubit QAOA shape, not a claim that
every circuit, CPU, thread count, or larger state vector receives the same
speedup.

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

No new user option is required: supported CPU circuits select the fast path
automatically, while unsupported layouts keep the existing implementation.

## Reproduce

Build the optional native CPU extension, then run:

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads qaoa_path_maxcut --n-wires 22 --layers 1 --threads 4 \
  --engines flagquantum_adjoint \
    flagquantum_adjoint_forward_rzz_rotation_rollback \
  --warmup 5 --iterations 31 --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_forward_rzz_rotation_cpu_arm64_20260929.json
```

The rollback engine temporarily disables
`FQ_NATIVE_CPU_ADJOINT_RZZ_H_FUSION`,
`FQ_NATIVE_CPU_SHARED_RZZ_FORWARD_FUSION`,
and `FQ_NATIVE_CPU_FORWARD_SPECIALIZED_ROTATIONS`, then restores the caller's
environment. This keeps both engines in one process with rotated execution
order and otherwise identical FlagQuantum code.
