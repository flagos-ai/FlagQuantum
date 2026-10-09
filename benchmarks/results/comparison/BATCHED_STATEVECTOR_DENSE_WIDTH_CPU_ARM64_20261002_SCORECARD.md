# Batched dense-fusion width CPU scorecard

## Conclusion

For the measured 18-qubit, batch-32 local-brickwork task, selecting six wires
per PyTorch dense-fusion group reduces the complete-task median from **1500.801
ms to 1088.262 ms**, a **1.379x speedup** over the focused four-wire rollback.
The same-run PennyLane Lightning bridge takes 1350.228 ms, so FlagQuantum is
**1.241x faster** on this exact task and machine.

Peak RSS also falls from **916.6 MiB to 655.9 MiB**, 28.4% below rollback.
PennyLane Lightning remains lower at 612.7 MiB, so FlagQuantum still uses 7.1%
more peak memory. This is a bounded Apple-arm64 result, not a universal
simulator ranking.

## What is measured and why it matters

The task returns 32 exact complex128 statevectors for independent parameter
bindings of one deterministic 18-qubit circuit. Each item contains 196 gates:
four full-width `RY`/`RZ` layers alternating with nearest-neighbor CX matchings,
followed by one batch-parameterized `RY` layer. Complete execution is timed
after construction and warmup, including result retrieval.

Before this change, each full-width dense rotation layer was partitioned into
`4 + 4 + 4 + 4 + 2` wires. The measured six-wire policy partitions it into
`6 + 6 + 6`, removing eight full-state traversals over the four layers. The
existing dense-matrix implementation and numerical order remain unchanged;
only the grouping width changes.

The policy is intentionally limited to complex128 CPU batches at 18 or 19
qubits. Five-sample selection probes found six wires faster at 18q batch 8/16/32
(323.671→236.586 ms, 711.549→401.968 ms, and 1009.138→789.483 ms) and at 19q
batch 8/16 (582.391→528.001 ms and 1178.428→1020.482 ms). At 20q batch 8 it
was slower (1306.251→1335.250 ms), so 20q and wider retain four-wire grouping.

## Formal results

All timing values are medians of eleven warm samples. Peak RSS is the median of
three fresh-process probes. Every timing series passes the 20% stability rule.

| Engine | Median task time | Relative result | Peak RSS | Execution RSS growth | Maximum state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| **FlagQuantum, six-wire adaptive width** | **1088.262 ms** | **1.379x faster than rollback** | **655.9 MiB** | **461.3 MiB** | 0 |
| FlagQuantum, four-wire focused rollback | 1500.801 ms | baseline | 916.6 MiB | 722.5 MiB | 7.70e-17 |
| **PennyLane Lightning bridge** | **1350.228 ms** | **FlagQuantum is 1.241x faster** | **612.7 MiB** | **417.5 MiB** | 6.94e-17 |

The optimized, rollback, and Lightning relative median absolute deviations are
15.88%, 15.98%, and 8.13%, respectively. Raw samples, versions, IR hash,
correctness values, and all memory probes are retained in the
[JSON artifact](batched_statevector_dense_width_cpu_arm64_20261002.json). The
generated [comparison table](https://github.com/FlagQuantum/FlagQuantum-evidence/releases/tag/evidence-2026-10-09.2)
is the compact view of the same run.

## User code

No public API changes are required. Eligible circuits select the measured width
automatically:

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(18, bsz=32, dtype=torch.complex128)
for layer in range(4):
    for wire in range(18):
        angle = 0.07 * (layer + 1) * (wire + 1)
        circuit.ry(wire, angle).rz(wire, -0.6 * angle)
    for left in range(layer % 2, 17, 2):
        circuit.cx(left, left + 1)

batch_angles = torch.linspace(-0.4, 0.4, 32, dtype=torch.float64)
for wire in range(18):
    circuit.ry(wire, batch_angles + 0.01 * wire)

states = circuit.state()  # shape: (32, 2**18)
```

Set `FQ_CPU_ADAPTIVE_DENSE_FUSION_WIDTH=0` to restore four-wire grouping.
Complex64, scalar batches, widths below 18 or above 19, non-CPU execution, and
the public circuit API are unchanged.

## Reproduction

From a source checkout with PennyLane extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_dense_width_rollback \
    pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-dense-width.json \
  --markdown-output BATCHED_DENSE_WIDTH.md
```

This is single-host, single-thread Apple-arm64 evidence. It is not a release
gate, multi-thread scaling result, distributed result, or claim about arbitrary
circuit families.
