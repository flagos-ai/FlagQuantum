# FlagQuantum IR Phase 3 Batch E implementation review

Updated: 2026-09-03

## Result

Batch E implements one private provider-neutral conformance harness and runs the exact
same ordered lifecycle checks against anonymous local runtime-plan, synthetic QASM text
and synthetic non-QASM artifact targets. It remains offline, absent from the stable
namespace and unreachable from `fq.run`, `fq.plan`, deployment and adapter paths.

The shared suite verifies:

- the sealed artifact before submission;
- submission and observable initial state;
- successful settling through the target-family driver;
- terminal success, result availability and terminal read idempotency;
- rejection of cancellation after successful completion;
- continuity across artifact, binding, receipt, result and conformance identities;
- explicit target-family, target-class, artifact-profile and adapter compatibility.

Provider extension metadata is immutable, dotted-namespace scoped and excluded from
semantic/execution identities. Reserved core keys, execution identity fields, account,
queue and quota state, credential-like keys, secret markers and URLs fail closed.

No provider SDK, network call, discovery, credential, real backend/job identity, new
emitter or remote submission is present.

## Evidence

- Authorization, three-family lifecycle, exact ordered-check, complete identity-chain,
  mismatch, extension privacy, immutability, determinism and private-namespace tests pass.
- Anonymous fixtures pin a conformance identity for each target family across Python hash
  seeds.
- The 10/100/1K/10K suite baseline is recorded in
  `tests/fixtures/internal_ir/phase3_batch_e_performance_baseline.json`.
- At 10K three-family suites (30K target lifecycles), observed p95 was 1,914.625838 ms
  and peak traced host memory was 37,078,803 bytes.

The measurements are environment-specific observations, not a public SLA. Proposed
private regression envelopes remain unapproved in
`tests/fixtures/internal_ir/phase3_batch_e_performance_budget.json`.

## Remaining closed surfaces

Batch E does not implement or authorize a real provider adapter, provider SDK, network
access, credentials, real remote execution, shadow/default integration, deployment
migration, public API changes, legacy retirement, or Batches F-H.

## Review decision requested

Approve only the private Batch E performance budget and its machine regression gate with:

```text
approve IR-PHASE3-BATCH-E-PERFORMANCE-BUDGET
```

Batch E exit and Batch F remain separately gated.
