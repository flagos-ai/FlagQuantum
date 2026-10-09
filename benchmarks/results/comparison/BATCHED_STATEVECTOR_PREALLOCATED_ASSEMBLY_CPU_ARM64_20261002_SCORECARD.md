# Preallocated CPU batch assembly vs rollback and PennyLane Lightning

## Conclusion

For four applicable 18-qubit, batch-32, complex128 inference tasks, writing each
completed CPU statevector window directly into its final tensor reduces median
execution RSS growth by **3.9% to 22.8%** against the exact functional-`cat`
rollback. Random Clifford also becomes **1.051x faster**; the other timing ratios
range from 0.974x to 1.010x, so the memory improvement costs at most 2.7% on this
run. FlagQuantum remains **1.100x to 2.416x faster** than PennyLane Lightning's
public native-batch path across all five measured tasks.

Local brickwork is deliberately excluded from preallocation. Its compiled
program combines batched fused rotation sequences with native Clifford matching;
an exploratory run showed that keeping the final 128 MiB output alive alongside
those workspaces increased peak memory. The selector therefore retains the
functional assembly for that structure. Differences between its two same-route
measurements are process and timing noise, not an optimization claim.

## What is measured and why it matters

Each call returns 32 independent exact statevectors with 262,144 complex128
amplitudes each. The logical result alone is 128 MiB. The prior implementation
retained every completed execution window and then allocated the complete result
for `torch.cat`, so both representations could be live at once. The selected
inference path allocates the result once and copies each completed window into
its final slice. Any window that requires gradients automatically keeps the
functional `cat` path so autograd semantics do not change.

Warm time is the median of 11 complete calls after two warmups. Peak RSS is the
median of three separate cold Python processes. Circuit construction and
one-time backend preprocessing are outside warm timing; state retrieval and
conversion to the common Torch result are included. Every raw timing and RSS
observation is retained in the adjacent JSON artifact.

## Measured results

`Rollback speedup` is rollback time divided by selected time. `FQ vs Lightning`
is Lightning time divided by FlagQuantum time. Values above one mean the selected
FlagQuantum path is faster.

| Workload | Selected FQ | Functional rollback | Rollback speedup | FQ vs Lightning | Selected execution RSS | Rollback execution RSS | RSS reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware efficient | 779.080 ms | 758.675 ms | 0.974x | **1.263x** | **359.4 MiB** | 465.4 MiB | **22.8%** |
| Truncated QFT | 787.166 ms | 794.837 ms | **1.010x** | **2.409x** | **363.3 MiB** | 378.1 MiB | **3.9%** |
| Random Clifford | **477.311 ms** | 501.866 ms | **1.051x** | **1.623x** | **334.7 MiB** | 394.2 MiB | **15.1%** |
| Local brickwork (excluded) | 1269.561 ms | 1235.449 ms | 0.973x | **1.100x** | 380.0 MiB | 355.7 MiB | same route; no claim |
| Dense nonlocal | 365.814 ms | 358.731 ms | 0.981x | **2.416x** | **354.6 MiB** | 386.3 MiB | **8.2%** |

All timing distributions satisfy the 20% relative-MAD stability rule. Selected,
rollback, and Lightning states agree within `1e-10`; the largest observed
absolute error is `8.689e-16`.

Total process peak RSS is 541.0 vs 658.8 MiB for Hardware efficient, 556.5 vs
547.4 MiB for Truncated QFT, 507.6 vs 569.8 MiB for Random Clifford, and 534.5
vs 553.9 MiB for Dense nonlocal. Because interpreter and allocator baselines
vary between fresh processes, the table uses baseline-adjusted execution growth
as the primary temporary-memory measure. The QFT total peak does not improve,
so this report makes no total-RSS claim for that case.

## FlagQuantum user code

No public API change is required:

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

The optimization applies only to chunked CPU inference. Inputs or parameters
that require gradients use functional assembly. Setting
`FQ_CPU_STATEVECTOR_BATCH_PREALLOCATED_ASSEMBLY=0` provides the complete
rollback used in this report.

## Reproduce

From a source checkout with PennyLane extras installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch flagquantum_native_assembly_rollback \
    pennylane_lightning_native_batch \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-preallocated-assembly.json \
  --markdown-output BATCHED_PREALLOCATED_ASSEMBLY.md
```

The complete machine-readable evidence is in the
[JSON artifact](batched_statevector_preallocated_assembly_cpu_arm64_20261002.json),
and the adjacent
[generated table](https://github.com/FlagQuantum/FlagQuantum-evidence/releases/tag/evidence-2026-10-09.2)
is a compact rendering of the same run.

## Limits and next action

- Single Apple-arm64 host, one CPU thread, Python 3.12.14, PyTorch 2.13.0,
  PennyLane 0.45.1, and PennyLane Lightning 0.45.0 only.
- Exact complex128 statevector inference only; no gradients, shots, GPU,
  distributed, arbitrary-width, or scalability claim.
- The local-brickwork selector is based on compiled program structure, not a
  workload name, and remains conservatively on the prior route.
- Process high-water RSS is noisy. Execution growth and all three raw probes are
  preserved; this is local non-release evidence.
- QFT, Random Clifford, Local brickwork, and Dense nonlocal still use more
  baseline-adjusted execution RSS than Lightning in this run. The next CPU PR
  should target kernel workspaces rather than batch-result assembly.
