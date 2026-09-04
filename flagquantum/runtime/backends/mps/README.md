# Distributed MPS Runtime

This directory owns rank-local state lifecycle, ownership and distribution,
transport sequencing, memory/workspace policy, checkpoint and restart,
production gates, profiling, and Runtime-facing result records.

It does not own tensor algebra or numerical kernels. Rank-local gate math lives
in `simulation/mps_rank_local.py`, compiled site kernels live in
`simulation/mps_site_kernels.py`, compiled layer contraction and factorization
live in `simulation/mps_compiled_layers.py`, and QR/SVD math lives in
`simulation/mps_factorization.py`. Rank-local adjoint projection and VJP
evaluation live in `simulation/mps_reverse.py`. Import these numerical owners
directly; local observable contractions live in
`simulation/mps_observables.py`. Do not recreate Runtime aliases or a second
implementation.

## Ten-minute change path

- Change rank ownership or communication order in `state.py`,
  `distribution.py`, or `communication.py`.
- Change forward ownership, communication, lifecycle, or evidence in
  `forward.py`; change reverse lifecycle in `reverse.py` or
  `reverse_replay.py`.
- Change rank-local adjoint projection or VJP evaluation in
  `simulation/mps_reverse.py`.
- Change local observable environment or MPO contraction math in
  `simulation/mps_observables.py`; change cross-rank scans and pipelines in
  `reverse_observables.py` or `reverse_z_observables.py`.
- Change memory admission or microbatch policy in `factorization.py`.
- Change numerical tensor behavior in the corresponding Simulation module,
  not here.

Run the focused MPS unit tests first, then the CPU vertical slice and
architecture checks. An ordinary numerical change should not require edits in
this directory.
