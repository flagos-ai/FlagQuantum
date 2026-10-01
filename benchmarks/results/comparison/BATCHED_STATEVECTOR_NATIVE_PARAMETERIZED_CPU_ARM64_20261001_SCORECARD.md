# Native batch-parameterized rotation layers on Apple arm64

## Conclusion

This measured change reduces the 18-qubit, batch-32 Random Clifford median from
908.163 ms to 801.939 ms, a **1.132x speedup** over the same FlagQuantum build
with only the new path disabled. It reduces local brickwork from 1078.723 ms to
941.103 ms, a **1.146x speedup**. Both comparisons use 11 interleaved timed
samples after two warmups.

PennyLane Lightning remains **1.766x faster** on Random Clifford (454.045 ms).
On local brickwork, FlagQuantum is **1.134x faster** than the measured Lightning
bridge (941.103 ms versus 1067.262 ms). This is a public-API task comparison:
Lightning executes the 32 independently parameterized circuits through the
existing FlagQuantum bridge, not through a framework-specific batched API.

## What is measured

Each task returns 32 exact complex128 statevectors with 18 qubits. Every circuit
item has 18 independent final RY parameters, for 576 independent parameters per
batch. Random Clifford contains 126 gates per item and local brickwork contains
196. Circuit construction and one-time backend setup are outside warm timing;
bridge conversion and result retrieval remain inside it. One CPU thread is used.

The optimization compiles consecutive disjoint RX, RY, or RZ gates into native
C++ layers of up to 11 wires. Their matrices may differ by batch row. The kernel
loads each local amplitude block once, applies the row-specific rotation
matrices, and writes the block once. It is enabled only for CPU inference batches
of at least two items. Autograd inputs or parameters, custom matrices,
unsupported devices, and builds without the native extension retain the existing
differentiable path.

## Same-run results

| Workload | Engine | Median task time | Relative result | Peak RSS | Maximum error vs optimized FQ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Clifford | FlagQuantum optimized | 801.939 ms | 1.132x faster than rollback | 710.5 MiB | 0 |
| Random Clifford | FlagQuantum rollback | 908.163 ms | baseline | 913.8 MiB | 4.25e-17 |
| Random Clifford | PennyLane Lightning bridge | 454.045 ms | 1.766x faster than FQ | 495.9 MiB | 2.78e-17 |
| Local brickwork | FlagQuantum optimized | 941.103 ms | 1.146x faster than rollback | 914.2 MiB | 0 |
| Local brickwork | FlagQuantum rollback | 1078.723 ms | baseline | 914.1 MiB | 6.25e-17 |
| Local brickwork | PennyLane Lightning bridge | 1067.262 ms | FQ is 1.134x faster | 558.0 MiB | 5.72e-17 |

All timing relative median absolute deviations are below 9.0%, and every
numerical comparison passes the 1e-10 absolute-error contract. Peak RSS is the
median of three fresh-process high-water measurements. Random Clifford peak RSS
falls by 22.3% versus rollback in this run; local-brickwork RSS is unchanged.
The RSS difference is supporting single-host evidence, not a universal memory
claim. Raw samples and environment metadata are retained in
[`batched_statevector_native_parameterized_cpu_arm64_20261001.json`](batched_statevector_native_parameterized_cpu_arm64_20261001.json).

## User code

No new public API is required. A normal batch-parameterized FlagQuantum circuit
selects the optimized path automatically when its applicability conditions hold:

```python
import flagquantum as fq
import torch

circuit = fq.Circuit(18, bsz=32)
parameters = torch.linspace(-0.2, 0.2, 32)
for qubit in range(18):
    circuit.h(qubit)
for qubit in range(0, 17, 2):
    circuit.cx(qubit, qubit + 1)
for qubit in range(18):
    circuit.ry(qubit, parameters + 0.01 * qubit)

states = circuit.state()  # shape: (32, 2**18)
```

Set `FQ_CPU_NATIVE_PARAMETERIZED_ONE_QUBIT_LAYER=0` to restore the previous
execution path for diagnosis or controlled comparison.

## Reproduction

From a source checkout with the native CPU extension built and the PennyLane
extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_parameterized_layer_rollback \
    pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-native-parameterized.json \
  --markdown-output BATCHED_NATIVE_PARAMETERIZED.md
```

This single-host Apple-arm64 result is comparison evidence, not a universal
framework ranking, release gate, multi-thread scaling result, or distributed
scalability claim.
