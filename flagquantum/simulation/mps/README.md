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
- Start in `static.py` for fixed-shape MPS programs and compiled VQE losses.
- Start in `tebd.py` for time-evolving block decimation workflows.
- Start in `dense_island.py` for experimental dense-island MPS numerics.
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

## Batched truncation

Samples share tensor shapes, so each bond retains the largest rank required
by any sample after applying `cutoff` and `max_bond`. Taking the smallest rank
would discard another sample's singular values above the cutoff. Independent
bonds in a factorization bucket still choose their own ranks. Discarded-weight
diagnostics report the largest squared discarded singular-value norm across
samples. Run `tests/unit/test_mps_batch_truncation.py` to check statevector
conversion, pair splitting, bucketed splitting, and projected gradients.

## Planning depends on the numerical state

`MPSPlanningMixin` is an abstract base for diagnostics and local refinement
planning. Its seven read-only properties describe state size, placement, and
bond dimensions. Its four numerical methods provide norms and canonical
residuals. `MPSState` supplies these implementations directly; there is no
adapter object, copied state, or second numerical implementation.

Keep numerical changes in `state.py` and planning decisions in `planning.py`.
When adding a planning dependency, declare it on the existing base and ensure
that the concrete state implements it. Do not model read-only properties as
mutable attributes merely to satisfy a checker. An incomplete numerical
implementation cannot be instantiated. Existing data-field declarations refer
to values initialized by `MPSState`; they do not create shared mutable defaults.

For this boundary, run `tests/test_mps.py` and the MPS initialization, noisy
lowering, and objective-pipeline tests under `tests/unit`. Preserve diagnostic
values, refinement decisions, and gradients. The abstract-base test verifies
that the base cannot be used as a standalone state and that a concrete Bell
state still produces a valid bond profile.
