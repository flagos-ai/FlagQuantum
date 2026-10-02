# Native CPU adjoint Euler post-reduction comparison

This report is generated from
[`native_cpu_adjoint_euler_post_reduction_cpu_arm64_20261002.json`](native_cpu_adjoint_euler_post_reduction_cpu_arm64_20261002.json).
It measures one exact complex128 weighted Z/ZZ expectation and the complete
gradient of every circuit parameter. Each median retains 21 samples, with
three value-and-gradient calls averaged per sample after five warmups, on Apple
arm64 with eight CPU threads. Circuit construction and optimizer updates are
excluded.

## Results

| Workload | Qubits | Post-reduction backward (ms) | Per-pair rollback backward (ms) | Backward speedup | Post-reduction value + gradient (ms) | Rollback value + gradient (ms) | Total speedup | PennyLane Lightning value + gradient (ms) | Lightning / FlagQuantum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 14 | 0.964 | 0.910 | 0.944x | 2.882 | 2.829 | 0.982x | 5.908 | 2.050x |
| Hardware-efficient VQE | 16 | 1.329 | 1.297 | 0.975x | 3.825 | 3.692 | 0.965x | 16.216 | 4.239x |
| Hardware-efficient VQE | 18 | 3.039 | 3.018 | 0.993x | 6.703 | 6.602 | 0.985x | 64.483 | 9.620x |
| Hardware-efficient VQE | 20 | 13.543 | 14.703 | **1.086x** | 26.177 | 27.423 | **1.048x** | 367.000 | 14.020x |
| Hardware-efficient VQE | 22 | 51.188 | 57.667 | **1.127x** | 92.716 | 100.833 | **1.088x** | 1798.152 | 19.394x |
| QAOA path MaxCut | 14 | 1.385 | 1.266 | 0.915x | 3.970 | 4.117 | 1.037x | 5.195 | 1.308x |
| QAOA path MaxCut | 16 | 2.312 | 2.433 | 1.053x | 5.615 | 5.639 | 1.004x | 13.140 | 2.340x |
| QAOA path MaxCut | 18 | 5.487 | 5.839 | 1.064x | 10.898 | 11.491 | 1.054x | 54.644 | 5.014x |
| QAOA path MaxCut | 20 | 23.592 | 24.307 | 1.030x | 44.937 | 45.483 | 1.012x | 327.587 | 7.290x |
| QAOA path MaxCut | 22 | 70.514 | 69.698 | 0.988x | 130.512 | 121.048 | 0.927x | 1234.762 | 9.461x |

Ratios above one mean the optimized FlagQuantum path is faster. The intended
workload is hardware-efficient VQE, whose RZ/RY/RX layer enters the Euler-triple
kernel. At 20 and 22 qubits, moving the coordinate transform after the SIMD
reduction improves backward by 1.086x and 1.127x and the complete operation by
1.048x and 1.088x. The 14--18-qubit cases are neutral to slightly slower in
this run, so this is a wide-state optimization rather than a universal latency
claim.

QAOA contains H, RX, and RZZ gates but no RZ/RY/RX Euler triples. Its two
FlagQuantum modes therefore execute the same production path and act as a
loaded-host noise control; the observed complete-call ratios span 0.927x to
1.054x. The maximum value-and-gradient relative median absolute deviation was
15.50%, below the benchmark's 20% stability limit.

PennyLane Lightning may perform most adjoint derivative work during QNode value
evaluation, so its framework-observed callback is not compared with
FlagQuantum backward. The method-matched cross-framework metric is the complete
value plus gradient. FlagQuantum is 1.308x to 19.394x faster on all ten measured
cases. This local, non-release result is not a universal framework ranking.

All cases passed the `1e-9` value/gradient tolerance. The maximum gradient error
was `1.78e-15` for the rollback and `1.77e-12` for PennyLane Lightning.

## What changed

The fused Euler-triple adjoint previously converted each amplitude pair's raw
X/Y/Z bilinears into RZ/RY/RX parameter coordinates inside the SIMD loop. That
repeated seven coefficient multiplications for every pair. The native C++
kernel now reduces the three raw bilinears first and applies the same linear
coordinate transform once after the reduction. It does not allocate another
state-sized buffer and does not change the public API.

Set `FQ_NATIVE_CPU_ADJOINT_EULER_POST_REDUCTION=0` to restore the exact per-pair
transform for diagnosis and same-binary A/B measurement.

## FlagQuantum example

No user-code change is required; the optimization is internal to the native CPU
adjoint path.

```python
import torch

import flagquantum as fq
from flagquantum import algorithms as fqa

torch.set_num_threads(8)
theta = torch.full((22, 3), 0.2, dtype=torch.float64, requires_grad=True)
circuit = fq.Circuit(22, dtype=torch.complex128)
for wire in range(22):
    circuit.rz(wire, theta[wire, 0])
    circuit.ry(wire, theta[wire, 1])
    circuit.rx(wire, theta[wire, 2])
for wire in range(21):
    circuit.cx(wire, wire + 1)
hamiltonian = fqa.Hamiltonian(
    tuple(fqa.pauli_term(0.7, "ZZ", (wire, wire + 1)) for wire in range(21))
    + tuple(fqa.pauli_term(0.2, "Z", (wire,)) for wire in range(22))
)
energy = hamiltonian.expectation(circuit, differentiation="adjoint")
energy.backward()
print(energy.item(), theta.grad)
```

## Reproduce

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 14 16 18 20 22 --layers 1 --threads 8 \
  --engines flagquantum_adjoint \
    flagquantum_adjoint_euler_post_reduction_rollback \
    pennylane_lightning_adjoint \
  --warmup 5 --iterations 21 --calls-per-sample 3 \
  --skip-memory-probe \
  --json-output native_cpu_adjoint_euler_post_reduction.json
```
