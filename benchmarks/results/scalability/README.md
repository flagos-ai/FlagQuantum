# Scalability Results

This directory is the only benchmark-results location for release-grade
scalability payloads. Every JSON file here must pass:

```bash
python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability
```

Accepted payloads must prove one logical workload sharded across ranks with
release evidence, capacity baseline failure details, sharded gradients,
sharded optimizer-update ownership, communication evidence, and empty blockers.
They must also use a signed `measured_production_run` envelope whose content,
raw-log checksum, commit, workload hash, command, devices, topology, rank
mapping, timings, memory, communication, warmup, iterations, seeds, and
fallback events were captured by execution. Unsigned JSON is rejected before
the semantic release gate runs.

Unsealed runs, capacity probes, failed promotion attempts, and raw logs belong
under `benchmarks/results/smoke/release_candidates/`. Moving a candidate into
this directory is a promotion action performed only after sealing and passing
the strict audit. An empty directory therefore means “no release-certified
scalability evidence”, not a successful certification.

The former CPU-labelled Phase 4 statevector payload is quarantined under
`benchmarks/fixtures/capacity_models/`. It is an estimated schema/capacity
fixture, not a promoted result, until reproduced on its claimed hardware.

## Phase 5 MPS Preparation

ISSUE-027 defined this contract before any MPS payload existed, and the heading
keeps that history. Two MPS payloads are promoted here now -- the capacity
completion and the matched-speed pair -- while the premise's first half, the
single-device capacity failure, is deliberately kept outside this directory as a
candidate, because one device has no ranks to shard across and the strict audit
would refuse it here. Read the list below as the requirements this directory's own
payloads satisfy rather than as a wish list. An MPS JSON in this directory
must include all general release fields plus:

- `state_mode` identifying an MPS path;
- `distribution_semantics="sharded_across_ranks"`;
- sharded MPS forward and executed production backward evidence;
- site, bond, boundary-gradient, and parameter-gradient ownership;
- executed boundary-adjoint exchange evidence;
- production-measured per-rank backward memory;
- production-executed per-boundary communication;
- sharded parameter, gradient, and optimizer-update ownership;
- `training_step_count > 0`;
- `single_gpu_expected_oom=True` with a real baseline device and failure
  reason;
- a scaling efficiency measured against the speedup the timed workload's own
  arithmetic permits rather than against the world size;
- explicit no-fallback semantics and empty blockers.

The single-device baseline is the capacity premise, so it has to describe the
frozen workload rather than a workload of its own: the gate re-reads the site
count, logical MPS size, bond dimension, peak memory, device capacity, workload
digest, workload body digest, and trainable parameter count against the manifest
and refuses a baseline that disagrees. The workload digest is the launcher that
pins the site count; the body digest is the module that builds the rank
boundaries and the parameterization, so a payload reports both, and a completion
payload that repeats a body digest this manifest never froze is refused as
evidence about a circuit nobody released. The
files the manifest cites for the premise are re-read and digested as well, so a
premise is established only while its sources resolve; a manifest that discloses
an absent source or a drifted artifact therefore reports
`capacity_premise_evidence_not_verifiable` until the files agree.

The scaling efficiency is a ratio, and what it is a ratio *of* is part of the
contract rather than the payload's choice. Dividing a measured speedup by the
world size is the efficiency of a fully partitionable workload; the timed MPS step
is not one, because two environment scans walk every site in a fixed global order,
so a share of its per-site cost is a recurrence no partition divides. A payload
therefore reports the serial fraction the frozen manifest names, the ceiling that
fraction implies at its own world size, the definition it divided by, and the
world-size ratio beside it as `linear_scaling_efficiency`. The gate recomputes the
ceiling from the frozen fraction and refuses a payload that names the replaced
definition, disagrees with the fraction, names a ceiling the fraction does not
imply, or reports an efficiency that is not the ratio of the two. The fraction is
not a constant in the manifest either: it is fitted by
`benchmarks/build_mps_shardability_calibration.py` and digest-bound to the
calibration artifact, so a manifest that names no artifact, an artifact that
moved, or an artifact stating a different fraction reports
`speed_scaling_calibration_not_verifiable` and the threshold is not evaluated.

Synthetic contract fixtures in pytest are schema tests only. They are not
benchmark results or public production evidence. A promoted MPS claim also
requires real accelerator or multi-node execution and must pass the audit
command above.

