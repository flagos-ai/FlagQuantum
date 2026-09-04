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

`mps.py` also resolves a trajectory seed into an explicit generator before it
invokes the Simulation callback. Noise lowering and MPS numerical evolution do
not belong here. `runtime/execution.py` receives the Compiler-lowered program
and supplies it to the MPS callback; the multi-trajectory loop never lowers the
same program per trajectory.

Run:

```bash
python -m pytest tests/unit/test_trajectory_runtime.py tests/test_noise.py -q
```
