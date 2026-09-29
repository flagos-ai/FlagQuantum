# Native CPU forward RZZ/rotation comparison

This report is generated from
[`native_cpu_forward_rzz_rotation_cpu_arm64_20260929.json`](native_cpu_forward_rzz_rotation_cpu_arm64_20260929.json).
It measures an exact 22-qubit QAOA path-MaxCut value-and-gradient evaluation
before and after the native CPU forward changes in this PR. Ratios above one
mean the optimized path is faster.

| Workload | Qubits | Gates | Parameters | Optimized forward (ms) | Rollback forward (ms) | Forward speedup | Optimized backward (ms) | Rollback backward (ms) | Backward speedup | Optimized total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QAOA path MaxCut | 22 | 65 | 2 | 50.465 | 97.692 | 1.94x | 93.862 | 94.408 | 1.01x | 144.316 | 192.909 | 1.34x | 1.776e-15 |

Both paths passed the `1e-9` correctness threshold. Thirty-one retained samples
followed five warmups. Total-time relative median absolute deviation was 1.49%
for the optimized path and 1.32% for rollback, both below the benchmark's 20%
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
also remove the legacy per-amplitude bit-test branch.
Contiguous butterfly pairs avoid a branch in the amplitude hot loop. Together,
these changes reduce full-state traffic and arithmetic without changing the
public API or the fallback path.

## Current level

On this Apple arm64 CPU with four PyTorch threads, the optimized FlagQuantum
forward pass takes 48.3% less time than the direct rollback of all three changes;
the complete value-and-gradient call takes 25.2% less time. This is useful evidence for the
22-qubit QAOA shape, not a claim that every circuit, CPU, thread count, or larger
state vector receives the same speedup. The unchanged backward kernel varied by
about 2% in this run and is slightly slower in the optimized samples, so the
forward improvement is the causal result while total time is the user-visible
outcome; its 0.6% movement is within ordinary run-to-run variation.

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
`FQ_NATIVE_CPU_SHARED_RZZ_FORWARD_FUSION`,
`FQ_NATIVE_CPU_FORWARD_SPECIALIZED_ROTATIONS`, then restores the caller's
environment. This keeps both engines in one process with rotated execution
order and otherwise identical FlagQuantum code.
