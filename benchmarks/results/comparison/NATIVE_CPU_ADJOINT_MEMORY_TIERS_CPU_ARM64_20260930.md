# Native CPU adjoint memory tiers

This report accompanies
[`native_cpu_adjoint_memory_tiers_cpu_arm64_20260930.json`](native_cpu_adjoint_memory_tiers_cpu_arm64_20260930.json).
It validates checkpoint-policy v5, which accounts for compact-cycle auxiliary
indices before selecting a CPU CNOT adjoint kernel.

## Result

| Selected tier | Planning budget | Modeled requirement | Full-state CX scratch | Auxiliary index bound | Forward | Backward | Value + gradient | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-auxiliary CNOT pairs | 1 GiB | 1 GiB | 0 | 0 | 263.302 ms | 573.243 ms | 846.828 ms | 1,283.406 MiB |
| Compact permutation cycles | 1 GiB + 144 MiB | 1 GiB + 144 MiB | 0 | 144 MiB | 244.107 ms | 414.805 ms | 665.454 ms | 1,285.188 MiB |
| Fused dual-state gather | 1.25 GiB | 1.25 GiB | 512 MiB | 0 | 240.156 ms | 403.472 ms | 648.038 ms | 1,475.297 MiB |

The planner now makes the time-memory tradeoff explicit. A strict 1 GiB
working-set budget selects the genuinely zero-auxiliary pair kernel. Adding the
cycle index bound selects a path that is **1.38x faster in backward** and
**1.27x faster end to end**. The cycle tier remains within 2.81% of fused
backward and 2.69% of fused total time while measured peak RSS is 190.109 MiB
lower.

The old planner counted four complex statevectors for both low-memory paths and
reported cycle metadata only after execution. Version v5 adds the conservative
nine-byte-per-amplitude bound before selection and exposes
`estimated_compact_reversible_bytes` plus `compact_cpu_cx_cycles` in the
execution summary. Short CNOT segments and disabled cycle kernels still select
the pair implementation without reserving cycle metadata.

## What is measured

The workload is one exact 24-qubit complex128 hardware-efficient VQE layer:
72 RX/RY/RZ parameters, a 23-gate nearest-neighbor CNOT chain, and 47 weighted
Z/ZZ Hamiltonian terms. Each retained sample evaluates the energy and obtains
all 72 gradients through Torch autograd and FlagQuantum's statevector adjoint.

All tiers produced energy `16.41756900554898` and full-gradient checksum
`-16.4751489020745`. Unit tests exercise both budget boundaries, verify the
selected tier in the public execution summary, and compare the value and every
parameter gradient with Torch autograd.

## Usage

Users set a budget; kernel selection remains automatic:

```python
from flagquantum.runtime.executors.statevector.reverse import (
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)

result = execute_torch_distributed_statevector_reverse(
    circuit,
    observable_wire=0,
    checkpoint_policy=StatevectorCheckpointPolicy(
        memory_budget_bytes=1_224_736_768,
    ),
)
result.backward()
print(result.summary()["checkpoint_policy"])
```

For this 24-qubit complex128 workload, `1_073_741_824` selects CNOT pairs,
`1_224_736_768` selects compact cycles, and `1_342_177_280` selects fused
gather. These thresholds scale with state size and dtype; they are not fixed
global constants.

## Reproduce

Compact cycles:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1224736768 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint --warmup 1 --iterations 6 \
  --calls-per-sample 1 --json-output compact-cycle-tier.json
```

Zero-auxiliary pairs:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1073741824 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint --warmup 1 --iterations 6 \
  --calls-per-sample 1 --json-output zero-auxiliary-pair-tier.json
```

Fused gather:

```bash
FQ_STATEVECTOR_CHECKPOINT_BUDGET_BYTES=1342177280 \
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe --n-wires 24 --layers 1 --threads 2 \
  --engines flagquantum_adjoint --warmup 1 --iterations 6 \
  --calls-per-sample 1 --json-output fused-gather-tier.json
```

Peak RSS values come from separate processes. The planning budget is a modeled
statevector and kernel-working-set boundary, not a hard operating-system RSS
limit. These are single-device Apple arm64 CPU measurements, not distributed
scaling or release-capacity evidence.
