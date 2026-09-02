# FlagQuantum IR Phase 3 Batch A implementation review

Updated: 2026-09-02

## Result

Batch A implements a private, provider-free `TargetCapabilities` boundary under
`flagquantum._compiler`. It is not exported through the stable package and is not used by
`fq.run`, `fq.plan`, deployment, adapters, or any default execution path.

The model now provides:

- closed target, artifact, result, control-flow and ancilla vocabularies;
- immutable gate, parameter-domain, topology and capability snapshots;
- deterministic canonical bytes and a SHA-256 semantic fingerprint;
- explicit logical/physical capacity, limits, calibration identity and validity metadata;
- structured, exhaustive fail-closed compatibility differences and diagnostics;
- anonymous local-runtime, QASM and non-QASM fixtures;
- rejection of provider, backend ID, credential, token, URL, job and queue fields.

Display labels are deliberately excluded from semantic identity. Provider identity,
credentials, account state, queue state and real backend IDs remain execution bindings and
cannot enter this snapshot.

## Evidence

- 30 focused authorization, schema, round-trip, identity, isolation, privacy,
  compatibility and namespace tests pass.
- Equal snapshots retain the same fingerprint across Python hash seeds.
- Topology, calibration, artifact and semantic capability changes affect identity.
- The 10/100/1K/10K capability-entry baseline is recorded in
  `contracts/ir-phase3-batch-a-performance-baseline.json`.
- At 10K entries, observed p95 was 3.442754 ms for construction plus fingerprint and
  14.506317 ms for compatible comparison; peak traced host memory was 1,531,880 bytes.

The measurements are environment-specific observations, not a public SLA. The proposed
private regression envelopes remain unapproved in
`tests/fixtures/internal_ir/phase3_batch_a_performance_budget_candidate.json`.

## Remaining closed surfaces

This batch does not implement or authorize TargetIR, ExecutableArtifact, RuntimeAdapter,
provider discovery, remote submission, conformance, shadow/default execution, public API
changes, legacy retirement, or Batches B-H.

## Review decision requested

Approve only the private Batch A performance budget and its machine regression gate with:

```text
approve IR-PHASE3-BATCH-A-PERFORMANCE-BUDGET
```

Batch A exit and Batch B remain separately gated.
