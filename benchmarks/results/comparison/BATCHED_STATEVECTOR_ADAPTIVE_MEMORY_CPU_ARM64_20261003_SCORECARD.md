# CPU batched statevector adaptive-memory scorecard

> Follow-up: the measured
> [`BATCHED_STATEVECTOR_QFT_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md`](BATCHED_STATEVECTOR_QFT_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md)
> extends the selector to controlled-phase graphs. Statements below that QFT
> retained 64 MiB describe this earlier measurement and its rollback snapshot.

## Conclusion

FlagQuantum now selects a smaller logical-state execution window only for the
two measured preallocated-output program families whose resident working set
benefits: CX-sequence programs and static/cross-wire diagonal graph programs.
On this Apple-arm64 profile, the 32 MiB window reduces execution RSS growth by
**41.5%** for Hardware Efficient and **23.8%** for Dense Nonlocal relative to
the fixed 64 MiB rollback. Hardware Efficient is also 1.041x faster; Dense
Nonlocal pays a bounded 2.8% timing cost. The fixed 64 MiB budget remains in
force for Truncated QFT, Random Clifford, Local Brickwork, direct assembly,
and program shapes outside the measured selector.

Across all five 18-qubit, batch-32, complex128 tasks, FlagQuantum remains
**1.283x to 2.303x faster** than PennyLane Lightning's public native-broadcast
path. Every state agrees within `1e-10`, and every 11-sample timing series
passes the 20% relative-MAD stability rule.

## What changed

The old policy used one 64 MiB logical-state budget for every bounded CPU
parameter batch. The new default retains that general budget and selects
32 MiB only when all of these conditions hold:

1. final-state assembly is preallocated rather than functional;
2. the compiled program contains a CX sequence, cross-wire diagonal region,
   or static CZ graph; and
3. `FQ_CPU_STATEVECTOR_BATCH_ADAPTIVE_BUDGET` has not disabled the policy.

This is a program-class decision, not a workload-name special case. The
runtime record exposes the selected budget and chunk size. Setting
`FQ_CPU_STATEVECTOR_BATCH_ADAPTIVE_BUDGET=0` restores the previous fixed
64 MiB behavior exactly.

## Measured results

Apple arm64, one CPU thread, Python 3.12.14, PyTorch 2.13.0, PennyLane 0.45.1,
and PennyLane Lightning 0.45.0. Timings are medians of 11 calls after three
warmups. RSS growth is the median of three fresh-process `ru_maxrss` probes.

| Workload | Adaptive FlagQuantum | Fixed-64 rollback | Adaptive / rollback | PennyLane Lightning | FlagQuantum speedup | Adaptive RSS growth | Rollback RSS growth | RSS reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware Efficient | **788.709 ms** | 821.285 ms | **1.041x faster** | 1014.999 ms | **1.287x** | **248.9 MiB** | 425.8 MiB | **41.5%** |
| Truncated QFT | 954.238 ms | 949.618 ms | 0.5% slower | 2197.745 ms | **2.303x** | 321.2 MiB | 319.2 MiB | -0.6% |
| Random Clifford | 386.300 ms | 356.223 ms | 8.4% slower | 772.229 ms | **1.999x** | **205.3 MiB** | 208.5 MiB | 1.5% |
| Local Brickwork | 1332.713 ms | 1281.532 ms | 4.0% slower | 1709.549 ms | **1.283x** | 310.7 MiB | **291.9 MiB** | -6.4% |
| Dense Nonlocal | 398.129 ms | **387.465 ms** | 2.8% slower | 886.504 ms | **2.227x** | **230.9 MiB** | 303.2 MiB | **23.8%** |

The three control workloads execute the same 64 MiB policy in both
FlagQuantum engines. Their observed differences therefore measure run-order,
allocator, and host noise rather than an algorithmic policy change. The
checked-in gate allows no control or target timing regression beyond 10%, and
requires at least 20% RSS reduction on both selected workloads. The memory
claim is made only for Hardware Efficient and Dense Nonlocal.

## User code

The optimization is automatic; users keep the native FlagQuantum API:

```python
import torch
import flagquantum as fq

angles = torch.linspace(-0.4, 0.4, 32, dtype=torch.float64)
circuit = fq.Circuit(18, bsz=32, dtype=torch.complex128)
for wire in range(18):
    circuit.ry(wire, angles + 0.01 * wire)
for left in range(17):
    circuit.cz(left, left + 1)

states = circuit.state()
assert states.shape == (32, 2**18)
```

## Reproduce

From a source checkout with the PennyLane extra installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_adaptive_budget_rollback \
    pennylane_lightning_native_batch \
  --threads 1 --warmup 3 --iterations 11 --memory-probes 3 \
  --json-output adaptive-memory.json \
  --markdown-output ADAPTIVE_MEMORY.md
```

The adjacent JSON contains every timing sample, every fresh-process RSS probe,
versions, workload hashes, correctness errors, and methodology fields.

## Limits and stop condition

- This is one Apple-arm64 host and one software profile; it is local comparison
  evidence, not a universal framework or scalability claim.
- The 18-qubit batch output alone is 128 MiB. This measures parameter-batch
  working-set behavior, not 22- or 26-qubit capacity.
- RSS is a process high-water mark. Median fresh-process deltas reduce startup
  bias but retain allocator and host noise, which is why no claim is made for
  the three unchanged controls.
- The optimization applies to exact CPU forward statevectors. It does not claim
  a backward, shot, GPU, or distributed-memory improvement.
- This PR stops when the two selected workloads each reduce median execution
  RSS by at least 20%, every timing remains within 10% of the fixed-budget
  rollback, all five workloads remain correct and stable, and FlagQuantum
  remains at least 1.25x faster than Lightning on this matched corpus.
