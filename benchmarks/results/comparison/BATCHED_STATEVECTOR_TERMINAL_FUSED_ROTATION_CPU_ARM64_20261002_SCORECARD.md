# Batched terminal fused-rotation CPU scorecard

## Conclusion

For the measured 18-qubit, batch-32 Dense nonlocal statevector task, routing
the terminal same-wire fused `RY` pairs through FlagQuantum's existing native
CPU layer kernel reduces the complete-task median from **421.472 ms to 373.666
ms**, a **1.128x speedup** over the exact rollback. PennyLane Lightning native
batch takes 807.468 ms in the same run, so FlagQuantum is **2.161x faster** on
this exact task and machine.

Peak RSS falls from **623.9 MiB to 567.8 MiB**, and execution RSS growth falls
from **458.0 MiB to 389.0 MiB**. PennyLane Lightning remains lower at 549.0
MiB peak RSS and 326.2 MiB execution growth. This is bounded Apple-arm64
evidence, not a universal framework ranking.

## What is measured and why it matters

The task returns 32 exact complex128 statevectors for independent parameter
bindings of one deterministic 18-qubit circuit. Each item contains 207 gates:
Hadamards on all wires, an all-to-all CZ graph, one fixed `RY` layer, and one
independently parameterized terminal `RY` layer. The fixed and terminal `RY`
on each wire form a same-wire fused sequence; before this change those
sequences were evaluated as dense PyTorch matrices rather than by the existing
native batch-specific one-qubit layer kernel.

Complete execution is timed after construction and warmup, including result
retrieval. The workload has 576 independent terminal parameters and its exact
statevector output alone occupies 128 MiB, so the result exercises both
full-state traversal cost and batch memory pressure.

## Formal results

All timing values are medians of 21 warm samples. Peak RSS is the median of
three fresh-process probes. Every timing series passes the 20% stability rule.

| Engine | Median task time | Relative result | Peak RSS | Execution RSS growth | Maximum state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| **FlagQuantum, native terminal fused layer** | **373.666 ms** | **1.128x faster than rollback** | **567.8 MiB** | **389.0 MiB** | 0 |
| FlagQuantum, fused-layer rollback | 421.472 ms | baseline | 623.9 MiB | 458.0 MiB | 5.64e-18 |
| **PennyLane Lightning native batch** | **807.468 ms** | **FlagQuantum is 2.161x faster** | **549.0 MiB** | **326.2 MiB** | 7.37e-18 |

The optimized, rollback, and Lightning relative median absolute deviations are
14.08%, 16.09%, and 18.82%, respectively. Raw samples, versions, IR hash,
correctness values, and all memory probes are retained in the
[JSON artifact](batched_statevector_terminal_fused_rotation_cpu_arm64_20261002.json).
The generated [comparison table](BATCHED_STATEVECTOR_TERMINAL_FUSED_ROTATION_CPU_ARM64_20261002.md)
is the compact view of the same run.

## Why the optimization is deliberately terminal-only

A broader experiment routed every eligible fused rotation region through the
native layer kernel. On 11-sample 18-qubit, batch-32 probes it made local
brickwork **9.8% slower** (959.486→1053.146 ms) and hardware-efficient
statevector **7.2% slower** (612.120→656.380 ms), while Dense nonlocal improved
by 1.037x (269.955→260.404 ms). The broader policy was rejected.

The shipped compiler therefore promotes only a final contiguous run of
same-wire `RX`/`RY`/`RZ` sequences, with at most six disjoint wires per native
layer. Intermediate fused regions retain their existing dense path. CPU
autograd, non-CPU execution, scalar batches, caller-owned input state, and the
public API remain unchanged. Set `FQ_CPU_NATIVE_FUSED_ROTATION_LAYER=0` for the
exact focused rollback.

## User code

No new API is required; eligible circuits select the path automatically:

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(18, bsz=32, dtype=torch.complex128)
for wire in range(18):
    circuit.h(wire)
for control in range(18):
    for target in range(control + 1, 18):
        circuit.cz(control, target)
for wire in range(18):
    circuit.ry(wire, 0.03 * (wire + 1))

angles = torch.linspace(-0.4, 0.4, 32, dtype=torch.float64)
for wire in range(18):
    circuit.ry(wire, angles + 0.01 * wire)

states = circuit.state()  # shape: (32, 2**18)
```

## Reproduction

From a source checkout with PennyLane extras installed:

```bash
pip install -e '.[pennylane]'

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_fused_rotation_layer_rollback \
    pennylane_lightning_native_batch \
  --threads 1 --warmup 3 --iterations 21 --memory-probes 3 \
  --json-output terminal-fused-rotation.json \
  --markdown-output TERMINAL_FUSED_ROTATION.md
```

This result does not cover backward/adjoint execution, multi-thread scaling,
GPU execution, other hosts, arbitrary circuit families, or release
scalability. Lightning's remaining memory advantage is explicit above.
