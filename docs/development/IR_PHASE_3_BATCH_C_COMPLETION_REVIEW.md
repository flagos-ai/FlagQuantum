# FlagQuantum IR Phase 3 Batch C completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- A closed private registry gives each supported artifact format an exact version, media
  type and encoding contract.
- Canonical runtime-plan JSON, static OpenQASM 2, static OpenQASM 3 and QCIS 1 payloads
  are validated against the complete `TargetIR`, not trusted by label or content hash.
- `SealedExecutableArtifact` is immutable and binds the source program, target program,
  target capability snapshot, compilation identity, profile, media type and payload hash.
- Seal and verify reject target/profile mismatch, noncanonical payloads, dynamic values
  unsupported by static text profiles, unsupported result requests and tampering.
- Provider identity, backend IDs, credentials, dispatch locators and mutable job state are
  excluded from the portable artifact boundary.
- Anonymous fixtures pin deterministic payload and artifact identities across supported
  local, QASM and non-QASM profiles.
- The approved private CPU budget is now an executable 10/100/1K/10K fail-closed gate.
- Stable public APIs, default execution, adapters and legacy behavior remain unchanged.

## Performance evidence

The approved seven-iteration, two-warmup Docker validation passed all latency, memory and
determinism thresholds. At 10,000 operations, sealing and verification measured
61.110471 ms p95 against a 180 ms budget, and peak traced host memory was 7,572,338 bytes
against a 33,554,432-byte budget.

These are private regression limits, not public service-level commitments.

## Batch D proposed boundary

Batch D may add only a private `RuntimeAdapter` protocol and provider-neutral execution
binding model after separate owner approval. Its required evidence is:

- `submit`, `status`, `cancel` and `result` lifecycle semantics with closed states;
- execution binding keeps immutable artifact identity separate from provider/job identity;
- local synchronous lifecycle conformance using an existing runtime-plan artifact;
- offline asynchronous mock lifecycle, including idempotent terminal-state behavior;
- result and receipt identity bind back to the submitted artifact without compiler rewrites;
- invalid transitions, unknown handles, duplicate terminal actions and identity mismatch
  fail closed with structured diagnostics;
- provider/backend/job fields appear only in execution bindings or receipts, never in
  `TargetIR`, compilation identity, portable artifact payloads or public evidence;
- independent performance baseline before any budget proposal;
- zero public/default-path impact.

Batch D does not authorize provider SDK calls, discovery, credentials, real remote
submission, new emitters, conformance across target families, shadow/default integration,
public API changes, legacy retirement, or Batches E-H.

## Requested decision

Approve Batch C exit and Batch D entry with:

```text
approve IR-PHASE3-BATCH-C-EXIT-BATCH-D
```
