# FlagQuantum IR Phase 3 Batch C implementation review

Updated: 2026-09-03

## Result

Batch C implements a private, immutable `SealedExecutableArtifact` boundary. It binds
already-emitted bytes to their source program, `TargetIR`, target-capability snapshot,
compilation identity and exact artifact profile without submitting or executing them.
It remains absent from the stable namespace and unreachable from `fq.run`, `fq.plan`,
deployment and adapter paths.

The implementation provides:

- a closed, versioned profile registry for canonical runtime-plan JSON, static OpenQASM
  2, static OpenQASM 3 and QCIS 1;
- exact payload-to-`TargetIR` validation instead of trusting a format label or hash;
- immutable bytes, explicit media type, content hash and deterministic artifact identity;
- target-profile acceptance and capability-fingerprint checks at seal and verify time;
- fail-closed rejection of mismatched programs, profiles, targets, identities, dynamic
  parameters, unsupported result requests and post-seal tampering;
- an artifact schema that excludes provider identity, real backend IDs, credentials,
  dispatch locations and job lifecycle state.

The profile code validates canonical encodings but is not a general-purpose emitter.
Adding an emitter remains separately gated work.

## Evidence

- Authorization, fixture, profile, exact-payload, identity, tamper, privacy, immutability,
  hash-seed and private-namespace tests pass.
- Four anonymous format fixtures have pinned content and artifact identities.
- The 10/100/1K/10K operation baseline is recorded in
  `tests/fixtures/internal_ir/phase3_batch_c_performance_baseline.json`.
- At 10K operations, observed seal-and-verify p95 was 51.503021 ms and peak traced host
  memory was 7,572,338 bytes for a 1,076,219-byte payload.

The measurements are environment-specific observations, not a public SLA. Proposed
private regression envelopes remain unapproved in
`tests/fixtures/internal_ir/phase3_batch_c_performance_budget.json`.

## Remaining closed surfaces

Batch C does not implement or authorize a new emitter, `RuntimeAdapter`, submission,
conformance, shadow/default execution, provider integration, real backend IDs, public API
changes, legacy retirement, or Batches D-H.

## Review decision requested

Approve only the private Batch C performance budget and its machine regression gate with:

```text
approve IR-PHASE3-BATCH-C-PERFORMANCE-BUDGET
```

Batch C exit and Batch D remain separately gated.
