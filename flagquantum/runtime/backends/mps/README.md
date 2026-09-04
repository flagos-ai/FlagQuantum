# Distributed MPS Runtime

This directory owns rank-local state lifecycle, ownership and distribution,
transport sequencing, memory/workspace policy, checkpoint and restart,
production gates, profiling, and Runtime-facing result records.

It does not own tensor algebra or numerical kernels. Rank-local gate math lives
in `simulation/mps_rank_local.py`, compiled site kernels live in
`simulation/mps_site_kernels.py`, and QR/SVD math lives in
`simulation/mps_factorization.py`. Import these numerical owners directly;
do not recreate Runtime aliases or a second implementation.

## Ten-minute change path

- Change rank ownership or communication order in `state.py`,
  `distribution.py`, or `communication.py`.
- Change forward/reverse lifecycle in `forward.py`, `reverse.py`, or
  `reverse_replay.py`.
- Change memory admission or microbatch policy in `factorization.py`.
- Change numerical tensor behavior in the corresponding Simulation module,
  not here.

Run the focused MPS unit tests first, then the CPU vertical slice and
architecture checks. An ordinary numerical change should not require edits in
this directory.
