# Cached Clifford phase maps on Apple arm64

## Conclusion

This measured change reduces the 18-qubit, batch-32 Random Clifford median from
673.693 ms to 605.827 ms, a **1.112x speedup** over the same FlagQuantum build
with only the new phase-map path disabled. Both comparisons use 11 timed samples
after two warmups.

Local brickwork contains no CZ gates, so this optimization is not applicable to
that workload. Its two FlagQuantum engines execute the same production path;
the observed 1197.026 ms versus 1110.186 ms difference is treated as host and
measurement variation, not as an optimization gain.

PennyLane Lightning remains **1.385x faster** on Random Clifford (437.393 ms).
Local brickwork is effectively at parity: Lightning is **1.002x faster**
(1107.697 ms versus 1110.186 ms). The optimization is therefore meaningful for
CZ-dense mixed Clifford matchings, but it is not a general statevector speedup
and does not close the Random Clifford framework gap.

## What is measured and why

Each task returns 32 exact complex128 statevectors with 18 qubits. Every circuit
item has 18 independent final RY parameters, for 576 independent parameters per
batch. Random Clifford contains 126 gates per item and local brickwork contains
196. Circuit construction and one-time backend setup are outside warm timing;
bridge conversion and result retrieval remain inside it. One CPU thread is used.

The preceding native mixed CX/CZ kernel still tested every CZ edge for every
destination amplitude. Focused measurements showed that four to seven CZ edges
made one matching kernel 1.99x to 5.15x slower than its CX-only inner loop. This
change folds each destination's CZ sign into the cached full CX gather map: a
negative encoded index means that the gathered amplitude must be negated. The
native loop now loads and decodes one integer rather than traversing all CZ
edges.

The signed map is used only when the existing full gather table is already
appropriate, currently below 22 qubits. At wider sizes the existing compact CX
representation and on-the-fly CZ masks remain in place, avoiding an exponential
index allocation. The signed-map cache is bounded at 128 MiB and evicts its
oldest entry. Autograd, non-CPU, scalar-batch, custom-matrix, and unavailable-
extension cases continue to use their existing safe paths.

## Same-run results

| Workload | Engine | Median task time | Relative result | Peak RSS | Maximum error vs optimized FQ |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Clifford | FlagQuantum optimized | 605.827 ms | 1.112x faster than rollback | 591.4 MiB | 0 |
| Random Clifford | FlagQuantum rollback | 673.693 ms | baseline | 589.0 MiB | 0 |
| Random Clifford | PennyLane Lightning bridge | 437.393 ms | 1.385x faster than FQ | 529.1 MiB | 2.78e-17 |
| Local brickwork | FlagQuantum optimized | 1110.186 ms | phase map not applicable | 861.9 MiB | 0 |
| Local brickwork | FlagQuantum rollback | 1197.026 ms | same production path; control only | 868.7 MiB | 0 |
| Local brickwork | PennyLane Lightning bridge | 1107.697 ms | 1.002x faster than FQ | 557.4 MiB | 5.72e-17 |

All timing relative median absolute deviations are below 12.7%, and every
numerical comparison passes the 1e-10 absolute-error contract. Peak RSS is the
median of three fresh-process high-water measurements. The Random Clifford
optimized path uses 2.4 MiB more peak RSS in this run, consistent with retaining
four signed int32 maps. Because local brickwork contains no CZ gates, both its
timing and RSS differences are control noise rather than evidence for this
change. Raw samples and environment metadata are retained in
[`batched_statevector_clifford_phase_map_cpu_arm64_20261001.json`](batched_statevector_clifford_phase_map_cpu_arm64_20261001.json).

## User code

No new public API is required. A normal batched circuit selects the cached phase
map automatically for eligible mixed matchings:

```python
import flagquantum as fq

circuit = fq.Circuit(8, bsz=32)
circuit.cx(0, 1)
circuit.cz(2, 3)
circuit.cx(4, 5)
circuit.cz(6, 7)

states = circuit.state()  # shape: (32, 2**8)
```

Set `FQ_CPU_NATIVE_CLIFFORD_PHASE_MAP=0` to retain the native mixed-matching
kernel while restoring its previous per-amplitude CZ-mask loop.

## Reproduction

From a source checkout with the native CPU extension built and the PennyLane
extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_clifford_phase_map_rollback \
    pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-clifford-phase-map.json \
  --markdown-output BATCHED_CLIFFORD_PHASE_MAP.md
```

This single-host Apple-arm64 result is comparison evidence, not a universal
framework ranking, release gate, multi-thread scaling result, or distributed
scalability claim.
