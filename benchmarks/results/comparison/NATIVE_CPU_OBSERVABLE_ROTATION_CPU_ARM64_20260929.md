# Native CPU observable/rotation boundary comparison

This report is generated from
[`native_cpu_observable_rotation_cpu_arm64_20260929.json`](native_cpu_observable_rotation_cpu_arm64_20260929.json).
It measures an exact 22-qubit QAOA path-MaxCut value-and-gradient evaluation
with the observable seed fused into the first reverse rotation tile, against a
direct rollback and PennyLane Lightning. Ratios above one mean FlagQuantum is
faster.

| Engine | Value evaluation (ms) | Autograd callback (ms) | Value + gradient (ms) | Callback ratio vs optimized | Value + gradient ratio vs optimized | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FlagQuantum fused observable/rotation | 74.265 | 69.109 | 141.595 | 1.00x | 1.00x | 0.000e+00 |
| FlagQuantum direct seed rollback | 76.461 | 70.938 | 145.618 | 1.03x | 1.03x | 0.000e+00 |
| PennyLane Lightning adjoint | 912.812 | 0.709 | 913.521 | 0.01x | 6.45x | 1.771e-12 |

All engines passed the `1e-9` correctness threshold. Thirty-one retained
samples followed five warmups. Total-time relative median absolute deviation
was 3.40% for the optimized path, 5.34% for rollback, and 3.21% for Lightning.
Value evaluation and autograd callback are framework-observed Torch phases,
not method-matched adjoint-kernel boundaries. Lightning may perform most of its
adjoint derivative work during QNode value evaluation. Only value + gradient is
comparable across engines; callback ratios are meaningful only between the two
FlagQuantum paths, which share an execution boundary.

## What is measured and why it matters

The workload prepares a 22-qubit superposition, applies a shared-parameter
nearest-neighbor RZZ cost layer and an RX mixer layer, evaluates the path-MaxCut
Hamiltonian, and computes the complete two-parameter adjoint gradient.

Previously, backward first traversed the final ket to write
`adjoint = 2 * H * ket`, then the native rotation kernel read both full states.
The optimized kernel generates each adjoint tile while loading the ket and
immediately evaluates and reverses the terminal rotations. This removes one
standalone full-state seed traversal without changing the public API.

## Current level

The isolated rollback shows a modest 1.03x backward improvement: median
backward time falls from 70.938 ms to 69.109 ms. The paired median saving is
2.032 ms and 23 of 31 paired samples favor the fused path. This is an
incremental memory-traffic improvement, not the 15% target originally
considered; the already-vectorized seed represented only a small share of the
22-qubit backward pass.

Lightning assigns most adjoint work to its measured forward phase, so its
0.709 ms backward number must not be compared in isolation. Under the shared
value-plus-gradient boundary, FlagQuantum takes 141.595 ms versus Lightning's
913.521 ms, or 6.45x less total time on this machine. These results apply only
to this workload, CPU, thread count, versions, and timing contract.

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

Supported local CPU circuits select the fused path automatically. Other
observables, devices, layouts, and non-terminal rotation shapes retain the
existing seed and reverse implementations.

## Reproduce

Build the optional native CPU extension, then run:

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads qaoa_path_maxcut --n-wires 22 --layers 1 --threads 4 \
  --engines flagquantum_adjoint \
    flagquantum_adjoint_observable_rotation_rollback \
    pennylane_lightning_adjoint \
  --warmup 5 --iterations 31 --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_observable_rotation_cpu_arm64_20260929.json
```

The rollback engine temporarily disables
`FQ_NATIVE_CPU_OBSERVABLE_ROTATION_BOUNDARY` and restores the caller's
environment. All other FlagQuantum code and optimization switches remain
identical.
