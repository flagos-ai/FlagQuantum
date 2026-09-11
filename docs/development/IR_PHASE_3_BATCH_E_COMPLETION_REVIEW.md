# FlagQuantum IR Phase 3 Batch E completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- One provider-neutral private conformance harness runs the identical ordered lifecycle
  contract for local runtime-plan, synthetic QASM text and synthetic non-QASM artifacts.
- Every family verifies artifact, submission, initial state, settling, terminal success,
  result identity, repeat-read idempotency and post-success cancellation rejection.
- Artifact, binding, receipt, result and conformance identities remain continuous and
  deterministic across Python hash seeds.
- Family, target class, artifact profile and runtime adapter incompatibility fails closed.
- Provider extensions are immutable, namespaced and explicitly non-semantic.
- Reserved identity fields, provider/backend/job fields, account/queue/quota metadata,
  credential-like keys, secret markers and URLs are rejected.
- Anonymous local, QASM and non-QASM fixtures pin deterministic golden identities.
- The approved private CPU budget is now an executable 10/100/1K/10K fail-closed gate.
- Stable public APIs, default execution, adapters and legacy behavior remain unchanged.

## Performance evidence

The approved seven-iteration, two-warmup Docker validation passed every latency, memory
and determinism threshold. At 10,000 three-family suites (30,000 target lifecycles),
observed p95 was 1,684.889368 ms against a 6,000 ms budget, and peak traced host memory
was 37,078,803 bytes against a 134,217,728-byte budget.

These are private regression limits, not public service-level commitments.

## Batch F proposed boundary

Batch F may add only a private, explicitly invoked shadow-comparison harness after
separate owner approval. Its required safety and evidence contract is:

- no environment-variable, import-hook, public API or default-path activation;
- an explicit immutable policy must enable each harness instance;
- legacy execution runs first and its result remains authoritative in every outcome;
- candidate exceptions, mismatches or budget breaches never replace or mutate the legacy
  result;
- closed mismatch taxonomy separates match, status, type/shape, value, identity and
  candidate-failure classes;
- a thread-safe kill switch blocks all future candidate work immediately;
- bounded comparison count, input/evidence size and measured overhead circuit breakers;
- privacy-safe evidence contains identities, hashes, sizes, taxonomy and bounded timings,
  never raw payloads, credentials, provider metadata or user data;
- deterministic offline fixtures cover match, mismatch, candidate failure, limit breach
  and kill-switch behavior;
- independent performance baseline before any budget proposal;
- zero public/default-path impact.

Batch F does not authorize production shadowing, background execution, telemetry export,
provider SDKs, network calls, deployment migration, public API changes, legacy retirement,
or Batches G-H.

## Requested decision

Approve Batch E exit and Batch F entry with:

```text
approve IR-PHASE3-BATCH-E-EXIT-BATCH-F
```
