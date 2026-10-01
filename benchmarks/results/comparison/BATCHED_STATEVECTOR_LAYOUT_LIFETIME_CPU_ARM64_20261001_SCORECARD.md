# CPU batched statevector layout-lifetime scorecard

## Decision

Pass for the measured 18-qubit, batch-32, complex128 CPU inference scope.
After a general dense gate's batched matrix multiplication completes,
FlagQuantum now releases the materialized input layout before allocating the
inverse-layout result. This removes one live 64 MiB execution-window tensor
without changing the matrix multiplication or any output bit.

This is a peak-memory change, not a claimed throughput optimization. The warm
median differs from exact rollback by less than 0.4% on both targeted workloads.

## What the benchmark measures

Each task evaluates 32 independent parameter bindings and returns all 32 exact
18-qubit complex128 statevectors. Execution uses 64 MiB batch windows on one CPU
thread. Warm time is the median of 11 complete calls after two warmups, with the
two engines rotated in execution order. Peak RSS is the median of three fresh
processes per engine and workload; every raw observation remains in JSON.

`Legacy layout retention` is an exact rollback that keeps the materialized
input layout alive while the inverse-layout output is allocated. All other
statevector optimizations are identical.

## Same-run result

Values above one in `Legacy/new` mean the new path is faster. RSS reduction is
computed from fresh-process median peak RSS.

| Workload | New time (ms) | Legacy time (ms) | Legacy/new | New peak RSS (MiB) | Legacy peak RSS (MiB) | RSS reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Clifford | 1049.385 | 1053.283 | 1.004x | 941.9 | 1005.8 | 6.4% |
| Local brickwork | 1061.349 | 1057.404 | 0.996x | 894.5 | 977.6 | 8.5% |

Both paths produce bitwise-identical statevectors. Relative timing MAD is at
most 5.4%, below the 20% contract limit. The result therefore supports the
memory claim while treating throughput as unchanged.

## Prominent external-framework context from #305

The stable #305 artifact used the same Apple arm64, Python 3.12, PyTorch 2.13,
complex128, one-thread, 18-qubit, batch-32 profile. These unchanged values show
the user-visible target; they were not rerun and are not used in this PR's
same-run ratios.

| Workload | #305 FQ budgeted time / RSS | Fastest external bridge time / RSS | External engine |
| --- | ---: | ---: | --- |
| Random Clifford | 1074.345 ms / 1167.6 MiB | 404.360 ms / 671.2 MiB | PennyLane Lightning |
| Local brickwork | 934.087 ms / 1042.8 MiB | 863.757 ms / 476.1 MiB | PennyLane Lightning |

External bridges execute 32 public single-item calls rather than each
framework's best private batching API. The present PR narrows FlagQuantum's
peak-RSS gap, but Random Clifford throughput remains substantially behind
PennyLane Lightning and is still an explicit optimization target.

## Reproduce

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_layout_retention \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-layout-lifetime.json \
  --markdown-output BATCHED_LAYOUT_LIFETIME.md
```

## Boundaries

- Gradient-tracked execution keeps the layout alive through its autograd graph,
  so this change does not reduce backward memory.
- A layout that is already a view has no materialized input allocation to free.
- Peak RSS includes interpreter/framework startup and circuit construction and
  is local comparison evidence, not a universal scalability claim.
- The optimization does not address the remaining Random Clifford throughput
  gap; it only removes avoidable live memory around the existing fast BMM.
