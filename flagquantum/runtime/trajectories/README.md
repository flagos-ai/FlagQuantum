# Trajectory Runtime

This directory owns trajectory identity, deterministic random streams,
rank-local work ownership, online statistics, stopping policy, failure handling,
checkpoint/restart, and cross-rank result aggregation.

It does not implement statevector or MPS gates, Kraus evolution, observable
kernels, noise lowering, or backend selection. Numerical trajectory executors
are supplied by Simulation; Runtime schedules them and records their outcomes.

## Ten-minute change path

- Change ID ownership in `ownership.py`.
- Change seed derivation in `rng.py`.
- Change online statistics in `statistics.py`.
- Change checkpoint schema or persistence in `checkpoint.py`.
- Change noisy MPS scheduling, recovery, or rank merging in `mps.py`.

Run:

```bash
python -m pytest tests/unit/test_trajectory_runtime.py tests/test_noise.py -q
```
