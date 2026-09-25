# Adaptive dense-fusion width on Apple arm64 CPU

This result measures exact complex128 statevector execution for circuits with
wide layers of independent single-qubit gates. FlagQuantum already combines
bounded wire-disjoint gates into a Kronecker product so that one matrix apply
streams the statevector for several gates. This change selects the product
width from measured state-size crossovers instead of using four wires below 22
qubits and six wires at every larger size.

The default policy now uses six wires at 18, 19, and 21 qubits, retains four at
the measured 20-qubit crossover, uses eight at 22 qubits, and retains six above
22. Batched states and complex64 keep the conservative four-wire bound. The
20- and 24-qubit exclusions are deliberate: paired probes found that six wires
slightly regressed representative 20-qubit layers, while eight regressed the
24-qubit hardware-efficient and local-brickwork workloads.

## Native rollback A/B

The run used one CPU thread, Python 3.12.14, and PyTorch 2.13.0. Each row used
two warmups and 15 retained end-to-end samples. `Off` restores the previous
four/six-wire policy with `FQ_CPU_ADAPTIVE_DENSE_FUSION_WIDTH=0`; `On` is the
new default. Every timing group passed the 20% relative median absolute
deviation threshold.

| Workload | Qubits | Off | On | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Hardware-efficient | 18 | 0.016924 s | 0.014003 s | 1.21x |
| Hardware-efficient | 22 | 0.315959 s | 0.287730 s | 1.10x |
| Local brickwork | 18 | 0.027315 s | 0.022509 s | 1.21x |
| Local brickwork | 22 | 0.588784 s | 0.563393 s | 1.05x |
| Dense nonlocal | 18 | 0.014406 s | 0.012066 s | 1.19x |
| Dense nonlocal | 22 | 0.287771 s | 0.258500 s | 1.11x |

This is a local CPU optimization result, not a universal framework ranking or
scalability claim.

## Refreshed cross-framework evidence

The adjacent corpus artifact remeasured FlagQuantum only and preserved the
Qiskit Aer, Cirq, and PennyLane payloads field-for-field. In its nine-sample
comparison, the relevant 22-qubit FlagQuantum medians are now 0.277011 seconds
for hardware-efficient, 0.503887 seconds for local brickwork, and 0.236885
seconds for dense nonlocal. These correspond to 1.72x, 1.42x, and 1.75x over
PennyLane Lightning on this host. See
`simulator_workload_corpus_cpu_arm64_20260924.json` for samples and correctness
evidence.

## User code

The optimization is automatic; user code does not change:

```python
import torch

import flagquantum as fq

circuit = fq.Circuit(22, dtype=torch.complex128)
for layer in range(4):
    for wire in range(22):
        angle = 0.07 * (layer + 1) * (wire + 1)
        circuit.ry(wire, angle)
        circuit.rz(wire, -0.6 * angle)
    for left in range(layer % 2, 21, 2):
        circuit.cx(left, left + 1)

state = circuit.state()
```

Reproduce either side of the native A/B by setting the switch to `0` or `1`:

```bash
FQ_CPU_ADAPTIVE_DENSE_FUSION_WIDTH=1 flagquantum-benchmark run simulator_workload_corpus \
  --workloads hardware_efficient_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 22 --engines flagquantum_native --threads 1 \
  --warmup 2 --iterations 15 --calls-per-sample 1 \
  --json-output adaptive-dense-width.json
```
