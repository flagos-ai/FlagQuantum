# CPU batched statevector allocation scorecard

## Decision

Pass for the measured 18-qubit, batch-32, complex128 CPU inference scope.
FlagQuantum now avoids materializing a full zero-state input before executing
64 MiB windows and writes inference-only one-qubit results directly into their
single output tensor. All five workloads remain correct, all timing MAD values
are at most 20%, and median peak RSS falls on every workload.

This is not a backward/adjoint improvement. Any state or gate matrix requiring
gradients keeps the functional autograd path.

## What the benchmark measures

Each task evaluates 32 independent parameter bindings of the same 18-qubit
circuit and returns all 32 exact complex128 statevectors. The logical output is
128 MiB. Warm time is the median of 11 complete calls after two warmups. Peak
RSS is the median of three separate cold Python processes; the JSON retains all
raw samples. One complete hardware-efficient case was rerun with three warmups
after monotonic host-load drift made the original timing distribution unstable;
the complete case, rather than selected samples, was replaced and recorded in
`methodology.focused_reruns`.

## Same-run result: new allocation path versus exact rollback

`Legacy` keeps 64 MiB execution windows but restores the former full zero-state,
functional one-qubit temporaries, and final `torch.cat` behavior. Values above
one in `Legacy/new` mean the new path is faster.

| Workload | New time (ms) | Legacy time (ms) | Legacy/new | New peak RSS (MiB) | Legacy peak RSS (MiB) | RSS reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient | 898.942 | 884.052 | 0.98x | 655.2 | 762.1 | 14.0% |
| Truncated QFT | 876.608 | 1084.592 | 1.24x | 654.8 | 774.9 | 15.5% |
| Random Clifford | 1394.936 | 1448.064 | 1.04x | 824.1 | 859.1 | 4.1% |
| Local brickwork | 1182.160 | 1144.981 | 0.97x | 859.2 | 882.5 | 2.6% |
| Dense nonlocal | 660.327 | 653.355 | 0.99x | 578.9 | 587.8 | 1.5% |

The memory win is consistent; throughput is materially better for Truncated
QFT, modestly better for Random Clifford, and within 3.2% of rollback for the
other three workloads. The host was busier than during the earlier #305 run, so
only same-run ratios—not absolute times across the two runs—support performance
claims here.

## Prominent external-framework context from #305

The stable #305 artifact used the same Apple arm64, Python 3.12, PyTorch 2.13,
complex128, one-thread, 18-qubit, batch-32 profile. These unchanged values show
the remaining user-visible gap, but are historical context rather than inputs to
the same-run speedup ratios above.

| Workload | #305 FQ budgeted time / RSS | Fastest external bridge time / RSS | External engine |
| --- | ---: | ---: | --- |
| Hardware-efficient | 813.310 ms / 1042.5 MiB | 727.017 ms / 637.4 MiB | PennyLane Lightning |
| Truncated QFT | 863.867 ms / 1166.1 MiB | 915.210 ms / 627.4 MiB | Cirq Simulator |
| Random Clifford | 1074.345 ms / 1167.6 MiB | 404.360 ms / 671.2 MiB | PennyLane Lightning |
| Local brickwork | 934.087 ms / 1042.8 MiB | 863.757 ms / 476.1 MiB | PennyLane Lightning |
| Dense nonlocal | 439.069 ms / 1042.0 MiB | 530.593 ms / 671.7 MiB | PennyLane Lightning |

The external bridges execute 32 public single-item calls; this is the task as
FlagQuantum exposes it today, not each framework's best private/native batching
API. The new allocation path closes part of the RSS gap, but Random Clifford and
Local brickwork still leave a concrete memory and throughput target.

## Reproduce

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_memory \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
    flagquantum_native_functional_windows \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output batched-memory-preallocation.json \
  --markdown-output BATCHED_MEMORY_PREALLOCATION.md
```

Use `FQ_CPU_STATEVECTOR_BATCH_BOUNDED_INITIAL_STATE=0` to restore the full
zero-state allocation and `FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT=0` to restore
functional one-qubit temporaries independently.

## Boundaries and next target

- Custom input statevectors are user-owned and remain fully materialized.
- Gradient-tracked execution deliberately retains the functional autograd path.
- The final `torch.cat` still coexists briefly with all completed windows; direct
  assembly was measured and rejected because it slowed four of five workloads.
- Peak RSS includes interpreter/framework startup and circuit construction and
  is local comparison evidence, not a universal scalability claim.
- The next memory PR should target final output assembly without its measured
  6%–17% throughput regression, or reduce the remaining fused-region temporary
  tensors on Random Clifford and Local brickwork.
