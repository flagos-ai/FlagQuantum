# Native CPU forward wide rotation-tile comparison

This report is generated from [`native_cpu_forward_wide_rotation_tiles_cpu_arm64_20260929.json`](native_cpu_forward_wide_rotation_tiles_cpu_arm64_20260929.json). It compares
FlagQuantum's eight-wire forward rotation tiles with the previous six-wire tiles.
Both paths use the same exact statevector-adjoint differentiation method; only
the forward tile width changes, and values plus full gradients must match.

| Workload | Qubits | Forward rotations | Wide forward (ms) | Six-wire rollback (ms) | Forward speedup | Wide total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 22 | 66 | 57.310 | 58.517 | 1.02x | 188.859 | 190.510 | 1.01x | 0.000e+00 |
| QAOA path MaxCut | 22 | 22 | 95.121 | 100.210 | 1.05x | 207.266 | 212.395 | 1.02x | 0.000e+00 |

## Interpretation

At 22 qubits, an eight-wire tile applies a complete 22-wire rotation layer in
three full-state passes instead of four six-wire passes. QAOA benefits most
because its forward RX layer is a larger share of total runtime. VQE remains
backward-dominated, so its end-to-end gain is smaller. This is non-release,
single-device evidence and is not a scaling claim.

## What is measured and why it matters

Each retained sample performs an exact expectation-value evaluation and the
full adjoint gradient. Hardware-efficient VQE exercises 66 RX/RY/RZ gates
and a nearest-neighbor CNOT chain; QAOA exercises an initial Hadamard layer,
21 shared-parameter RZZ gates, and 22 RX gates. This measures realistic
training calls rather than an isolated gate microbenchmark.

## Current level

On the recorded Apple arm64 process with two PyTorch threads, both workloads
improve in forward and complete value-plus-gradient time. The 1.01x-1.05x
range is an incremental memory-traffic improvement, not a claim that this
change closes the remaining gap to every external simulator or thread count.

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
    fqa.HamiltonianTerm(-0.5, {wire: 'z', wire + 1: 'z'})
    for wire in range(5)
)
energy = hamiltonian.expectation(circuit, differentiation='adjoint') + 2.5
energy.backward()
print(energy.item(), parameters.grad)
```

No new user option is required; eligible CPU rotation layers select the
eight-wire tile automatically.

## Reproduce

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 22 \
  --engines flagquantum_adjoint flagquantum_adjoint_forward_wide_tile_rollback \
  --layers 1 --threads 2 \
  --warmup 5 --iterations 41 \
  --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_forward_wide_rotation_tiles_cpu_arm64_20260929.json --markdown-output REPORT.md
```

Set `FQ_NATIVE_CPU_FORWARD_WIDE_ROTATION_TILES=0` for direct rollback.
