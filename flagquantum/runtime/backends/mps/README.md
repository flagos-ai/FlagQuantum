# Distributed MPS Runtime

This directory owns rank-local state lifecycle, ownership and distribution,
transport sequencing, memory/workspace policy, checkpoint and restart,
production gates, profiling, and Runtime-facing result records.

It does not own tensor algebra or numerical kernels. Rank-local gate math lives
in `simulation/mps_rank_local.py`, compiled site kernels live in
`simulation/mps_site_kernels.py`, compiled layer contraction and factorization
live in `simulation/mps_compiled_layers.py`, and QR/SVD math lives in
`simulation/mps_factorization.py`. Reverse pair factorization,
truncated-subspace projection, rank-local adjoint projection, and VJP
evaluation live in `simulation/mps_reverse.py`. Import these numerical owners
directly; local observable contractions live in
`simulation/mps_observables.py`. Do not recreate Runtime aliases or a second
implementation.
Canonicalization sweep order, ownership, transfers, and metrics remain in
`canonicalization.py`; its QR, transfer absorption, residual, and norm formulas
live in `simulation/mps_canonicalization.py`.

## Ten-minute change path

- Change rank ownership or communication order in `state.py`,
  `distribution.py`, or `communication.py`.
- Change canonicalization sweep ownership or transport in `canonicalization.py`;
  change its tensor math in `simulation/mps_canonicalization.py`.
- Change forward ownership, communication, lifecycle, or evidence in
  `forward.py`; change reverse lifecycle in `reverse.py` or
  `reverse_replay.py`. Their local instruction buckets call
  `simulation/mps_compiled_layers.py` rather than owning tensor contraction.
- Change reverse pair factorization, truncated-subspace projection, rank-local
  adjoint projection, or VJP evaluation in
  `simulation/mps_reverse.py`.
- Change local observable environment or MPO contraction math in
  `simulation/mps_observables.py`; change cross-rank scans and pipelines in
  `reverse_observables.py` or `reverse_z_observables.py`.
- Change memory admission or microbatch policy in `factorization.py`.
  That file also owns the bounded workspace pool and Runtime error translation;
  it must not wrap rank-local gate application or duplicate QR/SVD numerics.
- Change numerical tensor behavior in the corresponding Simulation module,
  not here.

Run the focused MPS unit tests first, then the CPU vertical slice and
architecture checks. An ordinary numerical change should not require edits in
this directory.
