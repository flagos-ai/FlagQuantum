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

ISSUE-027 defines the MPS release contract but does not add a promoted MPS
payload. A future MPS JSON in this directory must include all general release
fields plus:

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
- explicit no-fallback semantics and empty blockers.

Synthetic contract fixtures in pytest are schema tests only. They are not
benchmark results or public production evidence. A promoted MPS claim also
requires real accelerator or multi-node execution and must pass the audit
command above.

