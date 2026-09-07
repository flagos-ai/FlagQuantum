# MPS numerics

This package owns matrix-product-state numerical data and algorithms. It does
not select devices, schedule distributed work, manage jobs, or produce runtime
evidence.

- Start in `models.py` for MPS configuration, compiled local schedules, and
  numerical result records.
- Start in `state.py` for the batched MPS representation, local state
  operations, observables, and truncation diagnostics.
- Start in `factorization.py` for QR/SVD execution, rank selection, discarded
  weight, and factorization fallback behavior.
- Start in `local.py` for the local noiseless IR execution loop and instruction
  fusion.
- Start in `noisy.py` for one already-lowered noisy trajectory with an explicit
  random generator.
- Start in `planning.py` for bond profiles, adaptive bond growth, local
  refinement plans, and state summaries.
- Start in `rank_local.py` for rank-local instruction dispatch, gate
  application, and tensor sizing.
- Start in `site_kernels.py` for eager and compiled site kernels and their
  bounded compile cache.
- Start in `compiled_layers.py` for equal-shape instruction packing, batched
  contraction, and factorization.
- Start in `brickwork.py` for shape-bucketed nearest-neighbour sweeps.
- Start in `canonicalization.py` for canonical-site factorization, transfer
  absorption, residuals, and center norms.
- Start in `low_rank.py` for fixed-rank factorization primitives.
- Start in `observables.py` for local observable and MPO environment scans.
- Start in `reverse.py` for reverse factorization, projection, and local VJP
  primitives.
- Start in `entrypoints.py` for the stable MPS wrappers and Circuit/IR input
  adaptation; execution policy and trajectory lifecycle remain in Runtime.
- Import the owning submodule directly; this package does not re-export an MPS
  facade.
- Run `python -m pytest tests/test_mps.py -q` after a typical local change.
