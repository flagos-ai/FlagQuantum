# Native CPU adjoint host-memory budget

This report accompanies
[`native_cpu_adjoint_host_memory_budget_cpu_arm64_20260930.json`](native_cpu_adjoint_host_memory_budget_cpu_arm64_20260930.json).
It measures the 24-qubit performance cliff caused by applying a fixed 512 MiB
checkpoint budget to every CPU host.

## Result

| Policy | Selected strategy | Value evaluation | Autograd callback | Value + gradient | Peak RSS | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Host-aware default | reversible adjoint | 241.184 ms | 392.315 ms | 631.332 ms | 1,411.188 MiB | 5 stable samples passed |
| Static 512 MiB rollback | full rematerialization | — | >60 s | >60 s | — | timed out in first backward |

The new default is therefore **more than 95.0x faster** than the censored
rollback result. This is a lower bound, not an extrapolated completion time.
The five complete samples had 1.77% relative median absolute deviation.
This ratio measures removal of a checkpoint-policy cliff; it does not claim
that an individual statevector or adjoint kernel became 95x faster.

## What is measured and why it matters

The workload is one exact 24-qubit complex128 hardware-efficient VQE layer:
72 RX/RY/RZ parameters, a 23-gate nearest-neighbor CNOT chain, and 47 weighted
Z/ZZ Hamiltonian terms. Each retained sample evaluates the energy and obtains
all 72 gradients through Torch autograd and FlagQuantum's statevector adjoint.

A 24-qubit complex128 state occupies 256 MiB. The existing conservative
four-state model estimates a 1 GiB reversible working set. The former CPU
default always offered only 512 MiB, so 23 qubits fit exactly while 24 qubits
fell back to replaying the circuit before every differentiated gate. The new
policy obtains available physical memory from the CPU platform runtime, keeps
30% as headroom, and selects the fast path only when the estimate fits.

Linux cgroup v1/v2 limits cap the host result, so a container cannot mistake
the node's memory for its own allocation. Explicit
`FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES` and explicit checkpoint strategies
continue to override automatic planning. If memory is tight, the same safe
rematerialization fallback remains in place.

## FlagQuantum example

No new user option is required:

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

Measure the new default:

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 1 --iterations 5 --calls-per-sample 1 \
  --json-output host-aware.json
```

Restore the former static budget to reproduce the cliff. The recorded run was
interrupted after 60 seconds during its first backward:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=536870912 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 0 --iterations 3 --calls-per-sample 1 \
  --json-output static-512m.json
```

This is single-device Apple arm64 evidence. It is not a distributed scaling or
release-capacity claim, and hosts with insufficient available memory correctly
retain the slower bounded-memory fallback.
