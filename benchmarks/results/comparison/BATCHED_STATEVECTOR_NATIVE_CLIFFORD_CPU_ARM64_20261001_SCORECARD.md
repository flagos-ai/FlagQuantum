# Native static-Clifford batch layers on Apple arm64

## Conclusion

This measured change reduces the 18-qubit, batch-32 Random Clifford median from
1150.921 ms to 989.682 ms, a **1.163x speedup** over the same FlagQuantum build
with the new path disabled. It also reduces the local-brickwork median from
894.631 ms to 889.736 ms, a **1.006x ratio** that is too small to treat as a
meaningful improvement. Both comparisons use 11 alternating timed samples after
two warmups.

PennyLane Lightning remains **2.184x faster** on Random Clifford (453.091 ms),
so this result closes part of that gap rather than claiming parity. On the
measured local-brickwork task, Lightning is **1.027x faster** than FlagQuantum
(866.504 ms versus 889.736 ms). The bridge comparison answers the existing
public-API task; it is not a comparison with a native batched Lightning API.

## What is measured

Each task returns 32 exact complex128 statevectors with 18 qubits and independent
final RY parameters. Random Clifford contains 126 gates per item; local
brickwork contains 196. Circuit construction and one-time backend setup are
outside warm timing. External bridge conversion and result retrieval remain in
scope. One CPU thread is used throughout.

The optimization compiles consecutive parameter-free, disjoint H, S, S-dagger,
X, Y, and Z gates into native C++ layers of up to 11 wires. The kernel loads each
local amplitude block once, performs specialized Clifford arithmetic in place,
and writes the block once. It is enabled only for CPU inference batches of at
least two items. Gradients, parameterized gates, custom matrices, unsupported
devices, and builds without the native extension retain the existing path.

## Same-run results

| Workload | Engine | Median task time | Relative result | Peak RSS | Maximum error vs optimized FQ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Clifford | FlagQuantum optimized | 989.682 ms | 1.163x faster than rollback | 1004.1 MiB | 0 |
| Random Clifford | FlagQuantum rollback | 1150.921 ms | baseline | 942.7 MiB | 2.15e-17 |
| Random Clifford | PennyLane Lightning bridge | 453.091 ms | 2.184x faster than FQ | 557.5 MiB | 4.90e-17 |
| Local brickwork | FlagQuantum optimized | 889.736 ms | 1.006x vs rollback | 913.8 MiB | 0 |
| Local brickwork | FlagQuantum rollback | 894.631 ms | baseline | 913.9 MiB | 0 |
| Local brickwork | PennyLane Lightning bridge | 866.504 ms | 1.027x faster than FQ | 557.0 MiB | 8.44e-17 |

All timing relative median absolute deviations are below 12.1%, and all numerical
comparisons pass the 1e-10 absolute-error contract. Peak RSS is the median of
three fresh-process high-water measurements. Random Clifford peak RSS is 6.5%
higher than rollback in this run, while local brickwork is effectively
unchanged. The kernel is therefore accepted for the Random Clifford timing
improvement, not as a memory optimization or a general workload speedup. Raw
samples and environment metadata are retained in
[`batched_statevector_native_clifford_cpu_arm64_20261001.json`](batched_statevector_native_clifford_cpu_arm64_20261001.json).

## User code

No new public API is required. Existing FlagQuantum code selects the optimized
path automatically when its applicability conditions are satisfied:

```python
import flagquantum as fq
import torch

circuit = fq.Circuit(18, bsz=32)
for qubit in range(18):
    circuit.h(qubit)
for qubit in range(0, 17, 2):
    circuit.cx(qubit, qubit + 1)
parameters = torch.linspace(-0.2, 0.2, 32)
for qubit in range(18):
    circuit.ry(qubit, parameters)

states = circuit.state()  # shape: (32, 2**18)
```

Set `FQ_CPU_NATIVE_FIXED_ONE_QUBIT_LAYER=0` to restore the previous execution
path for diagnosis or controlled comparison.

## Reproduction

From a source checkout with the native CPU extension built and the PennyLane
extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_fixed_layer_rollback pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-native-clifford.json \
  --markdown-output BATCHED_NATIVE_CLIFFORD.md
```

This single-host Apple-arm64 result is comparison evidence, not a universal
framework ranking, release gate, multi-thread scaling result, or distributed
scalability claim.
