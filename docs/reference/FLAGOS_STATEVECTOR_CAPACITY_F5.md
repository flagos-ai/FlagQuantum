# FlagOS statevector capacity expansion (F5)

F5 is a matched, fail-closed development measurement of one statevector that
does not fit on a single A800 but completes when sharded across eight A800s
through FlagQuantum's public `backend="flagos"` and `device="flagos"` boundary.

The authoritative artifact is
[`artifacts/flagos_statevector_capacity_f5_a800_20260827.json`](../../artifacts/flagos_statevector_capacity_f5_a800_20260827.json),
with SHA-256
`128a6e7db8eee153d1bf1577d16ca55a38a7f7718d6da4661576ff9a759903cb`.

## Measured result

The exact matched workload is a 32-qubit, complex128, forward-only circuit. Its
logical statevector contains 2^32 amplitudes and requires 64 GiB of state
storage before runtime scratch space.

| Mode | World size | Result | Exact boundary |
| --- | ---: | --- | --- |
| Single device | 1 | Expected OOM measured | 64 GiB allocation failed on one A800; peak allocated bytes `68,719,477,248` |
| Replicated | 8 | Expected OOM on every rank | Each rank attempted the same full 64 GiB state and failed |
| Sharded | 8 | Passed | Each rank owned `536,870,912` amplitudes (8 GiB state); no full-state materialization |

Every sharded rank completed in approximately 1,910 seconds, reported zero
global norm error at the complex128 tolerance of `2e-11`, recorded 128
communication operations and `17,179,869,184` communication bytes, and stayed
below physical memory with peak allocated bytes `21,483,356,160` out of
`85,093,777,408` bytes.

The capacity run performs one full-width forward execution and validates its
global norm with FP64 scalar aggregation. It deliberately does not replay the
64 GiB logical workload: deterministic numerical behavior is covered by the
bounded F2 correctness cases, while F5 answers only the matched capacity
question. An initial two-replay protocol reached its 3,600-second subprocess
limit and was rejected; it is not counted as successful evidence.

## Evidence boundary

This result demonstrates development capacity expansion for this exact
single-node workload. It does not establish general scalability, performance,
convergence, backward or optimizer capacity, multi-node behavior, production
support, or release readiness.

The measured outer backend is FlagOS. The provider's inner communication route
was not independently identified, so the artifact records:

- `inner_communication_route = "unattributed"`;
- `flagcx_route_verified = false`;
- `host_staging_observed = null`;
- communication, scalability, production, and release claim flags as false.

The environment used FlagQuantum source revision
`348482a8f1278bcd406f66126f99bdf52755e6f5`, Torch-FL source revision
`1b19383c24d9aa54637dd23557a5a20ef6d52ec5`, PyTorch `2.11.0+cu130`, and
eight NVIDIA A800-SXM4-80GB devices. Torch-FL was built in an isolated
CUDA-boxing reference environment with FlagGems disabled.

## Reproduction and audit

The frozen manifest is
[`benchmarks/manifests/flagos_statevector_capacity_f5.json`](../../benchmarks/manifests/flagos_statevector_capacity_f5.json).
On an isolated eight-device FlagOS environment, run:

```bash
python tools/validate_flagos_statevector_capacity.py \
  --manifest benchmarks/manifests/flagos_statevector_capacity_f5.json \
  --output artifacts/flagos_statevector_capacity_f5_a800_20260827.json \
  --timeout-seconds 900
```

The controller runs single, eight-rank replicated, and eight-rank sharded modes
against the same canonical workload hash. The checked-in benchmark-contract
test reconstructs the profile from raw rank records and rejects mismatched
workloads, incomplete ranks, unmeasured OOMs, full-state materialization,
missing communication, invalid norm evidence, memory overflow, and provider
route overclaims.
