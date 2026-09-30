# Native CPU adjoint low-memory CX

This report accompanies
[`native_cpu_adjoint_low_memory_cx_cpu_arm64_20260930.json`](native_cpu_adjoint_low_memory_cx_cpu_arm64_20260930.json).
It measures a native in-place CPU CNOT sequence kernel that lets reversible
adjoint run when its faster fused-gather path does not fit the planning budget.

## Result

| Path | Planning budget | Full-state CX scratch | Value evaluation | Autograd callback | Value + gradient | Peak RSS | Relative total time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low-memory in-place CX | 1 GiB | 0 | 239.902 ms | 566.831 ms | 806.458 ms | 1,282.750 MiB | 1.25x |
| Fused CX gather | 1.25 GiB | 512 MiB | 239.428 ms | 404.246 ms | 643.363 ms | 1,474.781 MiB | 1.00x |
| #270 block checkpoints | 1 GiB | 0 | 1.213 s | 49.542 s | 50.756 s | not recorded | 78.89x |

At the same 1 GiB planning budget, the new route is **62.94x faster** than the
block-checkpoint fallback in value-plus-gradient time and **87.40x faster** in
the autograd callback. It eliminates all 2,304 replayed gates. In an isolated
process it also used 192.031 MiB less peak RSS than fused gather.

The tradeoff is explicit: applying the inverse CNOT sequence in place requires
one pass per CNOT rather than one fused permutation pass. It is 25.35% slower
than fused gather in warm total time, so the planner retains fused gather when
five state-equivalents fit and selects in-place CX only when four fit.

## What is measured and why it matters

The workload is one exact 24-qubit complex128 hardware-efficient VQE layer:
72 RX/RY/RZ parameters, a 23-gate nearest-neighbor CNOT chain, and 47 weighted
Z/ZZ Hamiltonian terms. Each retained sample evaluates the energy and obtains
all 72 gradients through Torch autograd and FlagQuantum's statevector adjoint.

The native C++ operator walks disjoint amplitude pairs for each inverse CNOT
and swaps both the ket and adjoint in place. It allocates only the small control
and target metadata tensors. The high-throughput route instead materializes a
new ket and adjoint, accounting for 512 MiB of reported full-state scratch at
24 qubits.

Both routes produced the same energy (`16.41756900554898`) and full-gradient
checksum (`-16.4751489020745`). Unit tests additionally compare every output
amplitude for complex64 and complex128, batch sizes one and two, and compare an
end-to-end value and all parameter gradients with native Torch autograd.

The planning budget describes statevector working-set selection. It is not a
hard operating-system RSS cap; the Python interpreter, Torch, shared libraries,
the forward state, and allocator retention are also present in measured RSS.

## FlagQuantum example

No API change is required. The planner selects the memory tier automatically:

```python
import torch

import flagquantum as fq
from flagquantum import algorithms as fqa

theta = torch.full((24, 3), 0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(24, dtype=torch.complex128)
for qubit in range(24):
    circuit.rx(qubit, theta[qubit, 0])
    circuit.ry(qubit, theta[qubit, 1])
    circuit.rz(qubit, theta[qubit, 2])
for qubit in range(23):
    circuit.cx(qubit, qubit + 1)

hamiltonian = fqa.Hamiltonian(
    fqa.pauli_term(0.7, "ZZ", (qubit, qubit + 1)) for qubit in range(23)
)
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduce

Measure the low-memory route:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1073741824 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 1 --iterations 4 --calls-per-sample 1 \
  --json-output low-memory-cx.json
```

Measure fused gather by allowing its five-state working set:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1342177280 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 1 --iterations 4 --calls-per-sample 1 \
  --json-output fused-cx.json
```

Restore the #270 block-checkpoint path:

```bash
FQ_NATIVE_CPU_CX_ADJOINT_INPLACE=0 \
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1073741824 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 0 --iterations 1 --calls-per-sample 1 \
  --json-output block-checkpoint-rollback.json
```

The warm medians use four retained samples after one cold run. Peak RSS comes
from separate one-iteration processes. These are single-device Apple arm64 CPU
measurements, not distributed scaling or release-capacity evidence.
