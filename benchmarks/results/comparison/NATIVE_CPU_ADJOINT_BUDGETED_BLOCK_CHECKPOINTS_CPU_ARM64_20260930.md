# Native CPU adjoint budgeted block checkpoints

This report accompanies
[`native_cpu_adjoint_budgeted_block_checkpoints_cpu_arm64_20260930.json`](native_cpu_adjoint_budgeted_block_checkpoints_cpu_arm64_20260930.json).
It measures the bounded-memory path used when a CPU adjoint workload cannot
retain the reversible working set but can retain a small number of block
checkpoints.

## Result

| Planning budget | Selected strategy | Checkpoints | Replayed gates | Value evaluation | Autograd callback | Value + gradient | Speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 GiB | budgeted blocks, interval 48 | 2 | 2,304 | 1.213 s | 49.542 s | 50.756 s | **1.84x** |
| 512 MiB rollback | full rematerialization | 1 | 4,560 | 1.283 s | 91.873 s | 93.155 s | 1.00x |

Under the measured 1 GiB planning budget, automatic planning retains the
initial state and one checkpoint after gate 48. That reduces executed replay
gates by 49.5% and total value-plus-gradient time by 45.5% compared with full
rematerialization. The planner reports both its estimated checkpoint count and
the backward pass reports the actual checkpoint and replay counts.

This is a fallback optimization, not a replacement for reversible adjoint. On
the same host, the reversible path completes warm samples in 0.654-0.712 s when
its estimated 1.25 GiB working set is allowed. The block path is selected only
for the narrow memory range where that fast path does not fit; it prevents the
former all-or-nothing jump directly to quadratic replay work.

## What is measured and why it matters

The workload is one exact 24-qubit complex128 hardware-efficient VQE layer:
72 RX/RY/RZ parameters, a 23-gate nearest-neighbor CNOT chain, and 47 weighted
Z/ZZ Hamiltonian terms. Each sample evaluates the energy and obtains all 72
gradients through Torch autograd and FlagQuantum's statevector adjoint.

A state occupies 256 MiB. For this CPU CNOT workload, planning reserves five
state-equivalents for reversible adjoint. A 1 GiB budget therefore cannot select
that path, but it can retain two checkpoints plus the adjoint and replay ket.
The interval is derived from the instruction count and available checkpoint
slots; it is not a workload-specific constant.

Small-circuit tests compare complex128 values and gradients with native Torch
autograd. They also assert the exact selected interval and replay count. The
24-qubit runs completed all 72 gradients and expose the measured execution
evidence above.

## FlagQuantum example

No API change is required. Automatic planning uses the host memory snapshot:

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

`FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES` remains an explicit planner input for
containers and reproducible experiments. It selects a bounded working-set
strategy but is not a hard operating-system limit on process RSS.

## Reproduce

Measure automatic block checkpointing with a 1 GiB planning budget:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1073741824 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 0 --iterations 1 --calls-per-sample 1 \
  --json-output block-1g.json
```

Measure the full-rematerialization rollback with a 512 MiB planning budget:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=536870912 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 0 --iterations 1 --calls-per-sample 1 \
  --json-output full-512m.json
```

The measurements are single-sample Apple arm64 CPU evidence because each
rollback sample takes about 93 seconds. They are not a distributed scaling or
release-capacity claim, and timings on other hosts will differ.
