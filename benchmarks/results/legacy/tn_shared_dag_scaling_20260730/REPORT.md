# TN shared-DAG scaling report

## Scope

Measured on one node with NVIDIA A800 80 GB GPUs.  The workload is a
40-qubit nearest-neighbour brickwork circuit of depth 4, requesting four
amplitudes.  It is deliberately a sparse-output TN workload; no full state is
materialized.

## Result

| GPUs | Before shared DAG (s) | Shared DAG (s) | Cold end-to-end (s) | Estimated per-slice cost |
|---:|---:|---:|---:|---:|
| 1 | 37.703 | 0.0937 | 9.90 | 14,720 |
| 2 | 37.145 | 0.1293 | 9.67 | 12,924 |
| 4 | 37.438 | 0.1079 | 8.70 | 12,024 |
| 8 | 37.835 | 0.1187 | 8.89 | 11,456 |

The shared contraction DAG removes repeated rank-local path construction and
improves single-GPU steady execution by about 402x.  Numerical amplitudes agree
across all world sizes to complex64 precision.

This workload does **not** strong-scale: one GPU is fastest in steady state.
The contraction is already only 94 ms, rank-local work falls by merely 22% from
1 to 8 GPUs, and process/NCCL overhead dominates.  Planning is currently about
7 seconds and dominates cold latency.

## Backend boundary

- Use SV for ordinary 30--35 qubit full-state or many-output workloads when it
  fits memory.
- Use MPS for circuits with low linear-cut entanglement and moderate bond
  dimension.
- Use single-GPU TN for low-treewidth, sparse-output circuits like this one.
- Use sliced distributed TN only when contraction width or peak memory requires
  it, or when each slice has enough arithmetic intensity to amortize launch and
  collective overhead.

The current selector and capacity tests encode this boundary.  GPU count alone
is not treated as a reason to distribute TN.

## Cotengra relationship

The implementation borrows the right architectural ideas from cotengra:
cost-aware multistart path search, explicit slicing, path reuse, and separating
planning from execution.  It is not a cotengra wrapper.  Earlier controlled
path-quality evidence is stored in `benchmarks/results/legacy/tn_cotengra_gap`.
Those results show parity on trivial networks, but cotengra remains stronger on
some non-trivial path-search cases.  Therefore path quality is still a
production gap even though the executor regression measured here is fixed.

## Production gate

Passed:

- Sparse outputs do not allocate a statevector.
- Shared contraction plans are reused across targets and slices.
- Cost-aware slicing produces balanced rank task assignment.
- 1/2/4/8-GPU results are persisted with phase timings and memory.

Open:

- Cache or broadcast planning so every rank does not repeat the same search.
- Add a minimum-work threshold before distributed TN is selected.
- Close the remaining cotengra path-quality gap on high-width 2D circuits.
- Demonstrate positive scaling on a memory-forced workload before calling the
  distributed executor production-ready.
