# Native disjoint CX/CZ matching on Apple arm64

## Conclusion

This measured change reduces the 18-qubit, batch-32 Random Clifford median from
989.075 ms to 754.687 ms, a **1.311x speedup** over the same FlagQuantum build
with only the new path disabled. It reduces local brickwork from 1165.631 ms to
972.286 ms, a **1.199x speedup**. Both comparisons use 11 timed samples after
two warmups.

PennyLane Lightning remains **1.454x faster** on Random Clifford (519.045 ms)
and **1.040x faster** on local brickwork (934.724 ms). The optimization therefore
closes part of the mixed-entangler gap but does not claim parity with Lightning.
This is a public-API task comparison: Lightning executes the 32 independently
parameterized circuits through the existing FlagQuantum bridge, not through a
framework-specific batched API.

## What is measured and why

Each task returns 32 exact complex128 statevectors with 18 qubits. Every circuit
item has 18 independent final RY parameters, for 576 independent parameters per
batch. Random Clifford contains 126 gates per item and local brickwork contains
196. Circuit construction and one-time backend setup are outside warm timing;
bridge conversion and result retrieval remain inside it. One CPU thread is used.

Random Clifford repeatedly creates disjoint layers containing both CX and CZ
gates. Previously, compilation split each logical matching into separate gate or
gate-kind regions, causing several complete statevector traversals. The new
inference-only C++ kernel composes the CX source-index mapping and CZ phase in
one output traversal. For the measured Random Clifford plan, four mixed layers
replace twenty entangler steps. A cached full CX index is used below 22 qubits;
the existing compact representation remains available at larger widths to avoid
an impractical index table.

The path activates only for CPU batches of at least two, exact built-in CX/CZ
instructions, pairwise-disjoint wires, no differentiable input/parameter/matrix,
and a loadable native extension. Unsupported cases keep the existing path.

## Same-run results

| Workload | Engine | Median task time | Relative result | Peak RSS | Maximum error vs optimized FQ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Clifford | FlagQuantum optimized | 754.687 ms | 1.311x faster than rollback | 588.7 MiB | 0 |
| Random Clifford | FlagQuantum rollback | 989.075 ms | baseline | 740.1 MiB | 0 |
| Random Clifford | PennyLane Lightning bridge | 519.045 ms | 1.454x faster than FQ | 557.6 MiB | 2.78e-17 |
| Local brickwork | FlagQuantum optimized | 972.286 ms | 1.199x faster than rollback | 866.2 MiB | 0 |
| Local brickwork | FlagQuantum rollback | 1165.631 ms | baseline | 914.0 MiB | 0 |
| Local brickwork | PennyLane Lightning bridge | 934.724 ms | 1.040x faster than FQ | 506.9 MiB | 5.72e-17 |

All timing relative median absolute deviations are below 17.6%, and every
numerical comparison passes the 1e-10 absolute-error contract. Peak RSS is the
median of three fresh-process high-water measurements. It falls by 20.5% versus
rollback on Random Clifford and 5.2% on local brickwork in this run. These RSS
differences are supporting single-host evidence, not universal memory claims.
Raw samples and environment metadata are retained in
[`batched_statevector_native_clifford_matching_cpu_arm64_20261001.json`](batched_statevector_native_clifford_matching_cpu_arm64_20261001.json).

## User code

No new public API is required. A normal batched circuit selects the optimized
path automatically when a disjoint mixed CX/CZ matching is encountered:

```python
import flagquantum as fq

circuit = fq.Circuit(8, bsz=32)
circuit.cx(0, 1)
circuit.cz(2, 3)
circuit.cx(4, 5)
circuit.cz(6, 7)

states = circuit.state()  # shape: (32, 2**8)
```

Set `FQ_CPU_NATIVE_CLIFFORD_MATCHING=0` to restore the previous execution path
for diagnosis or controlled comparison.

## Reproduction

From a source checkout with the native CPU extension built and the PennyLane
extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_clifford_matching_rollback \
    pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-native-clifford-matching.json \
  --markdown-output BATCHED_NATIVE_CLIFFORD_MATCHING.md
```

This single-host Apple-arm64 result is comparison evidence, not a universal
framework ranking, release gate, multi-thread scaling result, or distributed
scalability claim.
