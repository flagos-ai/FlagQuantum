# Native CPU adjoint compact CX cycles

This report accompanies
[`native_cpu_adjoint_cx_cycles_cpu_arm64_20260930.json`](native_cpu_adjoint_cx_cycles_cpu_arm64_20260930.json).
It measures an in-place CPU kernel that composes a CNOT segment into one compact
linear permutation and applies its disjoint cycles to the ket and adjoint.

## Result

| Path | Planning budget | Full-state CX scratch | Auxiliary index upper bound | Value evaluation | Autograd callback | Value + gradient | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Compact permutation cycles | 1 GiB | 0 | 144 MiB | 242.289 ms | 405.373 ms | 648.275 ms | 1,282.578 MiB |
| Per-CNOT pair rollback | 1 GiB | 0 | 0 | 249.264 ms | 522.795 ms | 771.345 ms | 1,282.656 MiB |
| Fused dual-state gather | 1.25 GiB | 512 MiB | negligible | 247.585 ms | 396.329 ms | 639.693 ms | 1,475.047 MiB |

The compact cycle route is **1.29x faster in backward** and **1.19x faster in
value plus gradient** than applying each CNOT as amplitude-pair swaps. It comes
within 2.28% of fused-gather backward time and 1.34% of fused total time while
using 192.469 MiB less measured peak RSS and no full-state CX scratch.

The 144 MiB number is a conservative worst-case bound: four bytes per amplitude
for the composed permutation, one byte per amplitude for cycle discovery, and
up to four bytes per amplitude for cycle leaders. This workload uses fewer
leaders. The planner budget selects statevector working sets; it is not a hard
process-RSS cap.

## What is measured and why it matters

The workload is one exact 24-qubit complex128 hardware-efficient VQE layer:
72 RX/RY/RZ parameters, a 23-gate nearest-neighbor CNOT chain, and 47 weighted
Z/ZZ Hamiltonian terms. Every retained sample evaluates the energy and obtains
all 72 gradients through Torch autograd and FlagQuantum's statevector adjoint.

The #272 low-memory kernel scanned the state once for every CNOT. This kernel
first composes the whole CNOT segment into a linear basis permutation, discovers
its disjoint cycles, and moves each ket and adjoint amplitude once around its
cycle. Independent cycles run in parallel. If the cycle path is unavailable or
disabled, the optimized pair kernel remains the safe zero-auxiliary fallback.

All three paths produced energy `16.41756900554898` and full-gradient checksum
`-16.4751489020745`. Direct kernel tests compare every amplitude for complex64
and complex128 with batch sizes one and two; the end-to-end test compares the
value and every parameter gradient with native Torch autograd.

## FlagQuantum example

No API change is required; the memory-aware adjoint planner selects the path:

```python
from itertools import chain

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
    chain(
        (
            fqa.pauli_term(0.7, "ZZ", (qubit, qubit + 1))
            for qubit in range(23)
        ),
        (fqa.pauli_term(0.2, "Z", (qubit,)) for qubit in range(24)),
    )
)
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduce

Measure compact permutation cycles:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1073741824 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 1 --iterations 6 --calls-per-sample 1 \
  --json-output compact-cx-cycles.json
```

Restore the per-CNOT pair kernel:

```bash
FQ_NATIVE_CPU_CX_ADJOINT_CYCLES=0 \
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1073741824 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 1 --iterations 8 --calls-per-sample 1 \
  --json-output per-cx-pairs.json
```

Measure fused gather by allowing its five-state planning tier:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1342177280 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint \
  --warmup 1 --iterations 6 --calls-per-sample 1 \
  --json-output fused-cx-gather.json
```

Peak RSS values come from separate processes. These are single-device Apple
arm64 CPU measurements, not distributed scaling or release-capacity evidence.
