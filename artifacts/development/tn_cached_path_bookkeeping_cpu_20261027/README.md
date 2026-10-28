# Tensor-network cached-path bookkeeping on CPU

A planned tensor-network contraction path is cached per network profile, so a
training loop over one network shape pays the path search once and then executes
the cached path. That steady-state execution re-derived the network's label
dimensions and label census from the active list at **every** pair step, which
makes the execution quadratic in the node count even though the path is already
known.

This artifact measures the change that derives both once per call and maintains
them where the loop changes them.

## What was measured

`benchmarks/tn_cached_path_throughput.py` contracts one operator network — the
matrix-product operator of a transverse-field Pauli sum over the ket and bra
copies of a depth-2 ansatz — after warming the path cache, and records the plan
shape beside the timing.

- **Before**: `origin/main` at `057424b5df734da9e14ca6e3400cdd93131d6cab`. The
  producer was copied into that checkout untracked to run it, so
  `measuring_source_committed` is `False` in those payloads; the script source is
  identical to the committed one, which is what `git diff` of the two runs'
  `workload` blocks confirms.
- **After**: `da8f27b43412fd1a36e0923423e0421ed89189f1` (this branch), with the
  producer committed, so `measuring_source_committed` is `True` in those payloads.

Both use three fresh processes per side, Python 3.12.14, PyTorch 2.14.1, macOS 27.0
arm64, CPU, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Rows below
are medians of the three processes, with each process's value beside it.

## Result

| Width | Nodes | Pair steps | Peak | Before (ms) | After (ms) | Speed-up | Per-process before | Per-process after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 117 | 116 | 1152 | 4.761 | 1.707 | 2.79x | 4.761, 4.940, 4.755 | 1.707, 2.046, 1.664 |
| 12 | 177 | 176 | 2176 | 10.316 | 2.576 | 4.00x | 10.316, 9.918, 11.693 | 2.576, 2.801, 2.569 |
| 14 | 207 | 206 | 2688 | 12.941 | 3.194 | 4.05x | 12.941, 13.740, 12.706 | 3.078, 3.218, 3.194 |

The contraction values are bit-identical before and after at every width:
`-4.106326232521626`, `-5.721513955374389`, `-6.701911766096943`.

The speed-up grows with the node count, which is the signature of removing a
quadratic term rather than shaving a constant: the before cost per pair step rises
from 41 µs at width 8 to 63 µs at width 14, while the after cost stays flat at
14.7–14.9 µs per pair step.

## What changed

`_contract_nodes_greedy` in `flagquantum/simulation/tensor_network/path_search.py`
now builds `dims` and `counts` once before the contraction loop, and updates
`counts` when a step removes its two operands and adds the produced node. The
per-step "cached path is stale" validation and the recorded
`PairContractionStep` provenance are unchanged, and the shape/dimension
consistency the initial scan performed is applied to each produced node, so no
check was removed — it changed from a whole-network rescan per step to a
per-node check per step.

Profiling the warm contraction before the change put 66% of the frame time in
`_label_dims` and `_label_counts` against 17% in the tensor arithmetic.

## Scope and remaining gaps

- Single-process CPU wall time of one internal contraction path. No accelerator,
  distributed, memory, time-to-target, or end-to-end training claim.
- The warm path is measured; the path search itself (paid once per new network
  shape) is not part of these numbers and is unaffected by this change.
- Three processes per side with medians. The spread is visible above: width 8
  after spans 1.66–2.05 ms, so the 2.79x is the smallest of the three speed-ups,
  not a best case.
- `dims` still keys on Python integers and is rebuilt per call. Making it a
  per-profile cached object was not attempted; the per-call cost is one linear
  scan, which the measurement above does not separate from the tensor work.
- No capability maturity changes. This is an internal execution-loop change with
  identical outputs.
