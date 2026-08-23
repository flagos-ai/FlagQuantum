# 36q capacity-forced checkpointed TN gradient

The workload is a 36-qubit 4x9 grid, four cycles, complex128 expectation
gradient with 36 parameters.  It uses 128 rank-owned slices, a 4 GiB
intermediate limit, 16 GiB checkpoint budget, and 64 GiB reverse working-set
budget.

| A800s | Slices/rank | Seconds | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 128 | 140.904 | 1.00x | 100% |
| 2 | 64 | 73.065 | 1.93x | 96% |
| 4 | 32 | 36.742 | 3.83x | 96% |
| 8 | 16 | 17.646 | 7.99x | 99.8% |

All runs produced finite outputs, parameter gradients, and cotangents.  Peak
allocated memory is 43.64 GiB/rank and peak reserved memory is 48.43 GiB/rank.
Each rank saves about 15.99 GiB of checkpoint tape.  No full-state or full-tape
fallback is used.

Communication is small: 37 collectives and 304 bytes/rank of gradient result
payload.  Scaling is driven by balanced rank-owned slice work.

This passes the single-node capacity, memory, finite-gradient, ownership, and
strong-scaling gates.  Numerical semantics are covered by separate
reference-sized distributed tests; a dense 36q complex128 reverse reference is
not materialized.
