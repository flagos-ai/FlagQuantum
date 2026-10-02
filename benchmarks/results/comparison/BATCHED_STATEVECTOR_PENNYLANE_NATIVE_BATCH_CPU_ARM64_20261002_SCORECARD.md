# FlagQuantum vs PennyLane Lightning native batch

## Conclusion

On this Apple-arm64, one-thread, complex128 profile, FlagQuantum completes all
five measured 18-qubit, batch-32 exact-statevector tasks faster than PennyLane
Lightning's public native broadcast path. The observed FlagQuantum speedup is
**1.229x to 2.273x**. This replaces the weaker comparison against 32 repeated
FlagQuantum-to-PennyLane bridge calls with a method that gives Lightning one
broadcasted circuit and performs its device preprocessing once.

This is local, non-release evidence for a finite corpus. It does not claim that
FlagQuantum is faster for arbitrary circuits, widths, batch sizes, thread
counts, devices, or PennyLane versions.

## What is measured and why it matters

Each case returns 32 exact complex128 statevectors for independent parameter
bindings of one circuit structure. Circuit construction and one-time backend
preprocessing are outside warm timing; execution, state retrieval, and conversion
to the common Torch result are included. Each timing is the median of 11 samples
after two warmups. Peak RSS is the median of three fresh-process probes.

The benchmark now distinguishes two PennyLane paths:

- **PennyLane Lightning native batch** uses public parameter broadcasting,
  preprocesses the broadcast expansion once, and executes the resulting batch
  directly on `lightning.qubit`.
- **PennyLane Lightning bridge** invokes the current single-item FlagQuantum
  ecosystem bridge 32 times. It measures today's interop user path, not
  Lightning's best batching capability.

## Measured performance

`FQ speedup` is Lightning native-batch time divided by FlagQuantum time. Values
above one mean FlagQuantum was faster.

| Workload | FlagQuantum | Lightning native batch | FQ speedup | Lightning bridge |
| --- | ---: | ---: | ---: | ---: |
| Hardware efficient | **582.761 ms** | 748.519 ms | **1.284x** | 808.712 ms |
| Truncated QFT | **592.609 ms** | 1346.926 ms | **2.273x** | 1406.096 ms |
| Random Clifford | **268.050 ms** | 370.231 ms | **1.381x** | 400.739 ms |
| Local brickwork | **682.513 ms** | 838.924 ms | **1.229x** | 928.575 ms |
| Dense nonlocal | **231.709 ms** | 431.296 ms | **1.861x** | 499.661 ms |

All 15 timing distributions satisfy the 20% relative-MAD stability rule. Every
Lightning native-batch state agrees with FlagQuantum within `1e-10`; the largest
observed absolute error is `8.689e-16`.

The checked performance gate covers all five workload hashes. It permits at
most 10% regression in FlagQuantum time or peak RSS and requires
`FlagQuantum / Lightning <= 0.90`, preserving at least a 1.111x native-batch
lead. All five cases pass; the measured ratios range from `0.440` to `0.814`.

## Memory result and remaining gap

FlagQuantum's total process peak RSS is lower in all five cases because its
framework/interpreter baseline is smaller:

| Workload | FQ peak RSS | Lightning peak RSS | FQ execution RSS growth | Lightning execution RSS growth |
| --- | ---: | ---: | ---: | ---: |
| Hardware efficient | **636.3 MiB** | 749.9 MiB | 442.2 MiB | **388.2 MiB** |
| Truncated QFT | **582.5 MiB** | 775.0 MiB | 388.8 MiB | **385.3 MiB** |
| Random Clifford | **583.4 MiB** | 785.4 MiB | 390.3 MiB | **379.4 MiB** |
| Local brickwork | **630.7 MiB** | 740.6 MiB | 437.2 MiB | **357.7 MiB** |
| Dense nonlocal | **639.3 MiB** | 726.6 MiB | 462.4 MiB | **342.4 MiB** |

The baseline-adjusted execution growth is therefore still higher for
FlagQuantum in every case, by about 1% to 35%. That is the actionable remaining
CPU batch gap. Local brickwork also has the smallest timing lead and should stay
in the regression corpus while temporary-memory work proceeds.

## FlagQuantum user code

No API change is required to use native parameter batching:

```python
import torch
import flagquantum as fq

angles = torch.linspace(-0.4, 0.4, 32, dtype=torch.float64)
circuit = fq.Circuit(18, bsz=32, dtype=torch.complex128)
for wire in range(18):
    circuit.ry(wire, angles + 0.01 * wire)

states = circuit.state()
assert states.shape == (32, 2**18)
```

The PennyLane native-batch path added here is benchmark-only. It does not widen
the public single-item `flagquantum.ecosystem.pennylane.run` contract.

## Reproduce

From a source checkout with PennyLane extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch pennylane_lightning_native_batch \
    pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-native-vs-lightning.json \
  --markdown-output BATCHED_NATIVE_VS_LIGHTNING.md

python -m flagquantum.benchmarking.cpu_performance_gate \
  benchmarks/results/comparison/batched_statevector_pennylane_native_batch_cpu_arm64_20261002.json \
  batched-native-vs-lightning.json \
  --max-slowdown 1.10 --max-memory-growth 1.10 \
  --minimum-samples 5 --minimum-memory-probes 3 \
  --comparison-engine pennylane_lightning_native_batch \
  --max-native-over-comparison 0.90 \
  --json-output batched-native-vs-lightning-gate.json
```

The complete raw samples, versions, methodology, IR hashes, correctness values,
and RSS probes are retained in the
[JSON artifact](batched_statevector_pennylane_native_batch_cpu_arm64_20261002.json).
The adjacent
[generated table](BATCHED_STATEVECTOR_PENNYLANE_NATIVE_BATCH_CPU_ARM64_20261002.md)
is a compact view of the same run. The
[gate verdict](batched_statevector_pennylane_native_batch_gate_cpu_arm64_20261002.json)
records the accepted profile, thresholds, and per-case results.

## Limits and next action

- Single host, one CPU thread, Apple arm64, Python 3.12.14, PyTorch 2.13.0,
  PennyLane 0.45.1, and PennyLane Lightning 0.45.0 only.
- Exact statevector output only; no gradients, shots, GPU, or distributed claim.
- Native-batch preprocessing is excluded for both engines; returned-state
  materialization is included.
- Process peak RSS includes different framework baselines, so execution growth
  is reported separately and is the more actionable temporary-memory signal.
- The next CPU PR should reduce FlagQuantum's batch execution RSS growth while
  preserving the measured time lead and exact-state correctness.
