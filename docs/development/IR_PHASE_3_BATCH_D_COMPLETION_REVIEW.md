# FlagQuantum IR Phase 3 Batch D completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- A provider-neutral private `RuntimeAdapter` protocol defines `submit`, `status`,
  `cancel` and `result` without accepting source IR or performing compilation.
- Closed execution states cover queued, running, succeeded, failed and cancelled.
- Immutable execution options, bindings, handles, receipts and results form a complete
  chain back to the submitted artifact identity.
- Provider/backend/job identity is explicitly execution-only and receipt-scoped; all
  delivered fixtures and adapters remain anonymous and offline.
- A local synchronous adapter consumes canonical runtime-plan artifacts and sanitizes
  executor exceptions.
- An offline asynchronous mock proves success, failure and cancellation transitions,
  including idempotent terminal behavior.
- Invalid artifacts/options, unknown or identity-tampered handles, conflicting terminal
  results and invalid transitions fail closed.
- Concurrent submissions allocate unique identity-bound handles under a lock.
- ABI result objects reject contradictory states such as success without a receipt,
  status or result.
- The approved private CPU budget is now an executable 10/100/1K/10K fail-closed gate.
- Stable public APIs, default execution, adapters and legacy behavior remain unchanged.

## Performance evidence

The approved seven-iteration, two-warmup Docker validation passed every latency, memory
and determinism threshold. At 10,000 complete local lifecycles, observed p95 was
424.923267 ms against a 1,000 ms budget, and peak traced host memory was 10,422,259 bytes
against a 67,108,864-byte budget.

These are private regression limits, not public service-level commitments.

## Batch E proposed boundary

Batch E may add only a private provider-neutral conformance harness after separate owner
approval. The identical suite must run against three anonymous target families:

- local runtime-plan;
- synthetic QASM text;
- synthetic non-QASM artifact.

Its required evidence is:

- the same artifact/runtime lifecycle assertions execute for all three families;
- each fixture preserves source, target, compilation, artifact, binding, receipt and
  result identity continuity;
- format/profile mismatches and incompatible target/adapter pairs fail closed;
- provider extensions use an explicit namespaced, immutable, non-semantic envelope;
- reserved core keys, credentials, secrets and mutable queue/account data are rejected;
- no provider SDK, network, discovery, remote submission or real backend/job identity;
- independent performance baseline before any budget proposal;
- zero public/default-path impact.

Batch E does not authorize a real provider adapter, shadow/default integration, deployment
migration, public API changes, legacy retirement, or Batches F-H.

## Requested decision

Approve Batch D exit and Batch E entry with:

```text
approve IR-PHASE3-BATCH-D-EXIT-BATCH-E
```
