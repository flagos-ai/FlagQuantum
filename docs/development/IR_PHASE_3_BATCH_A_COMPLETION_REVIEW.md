# FlagQuantum IR Phase 3 Batch A completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- A private, immutable and provider-free `TargetCapabilities` schema describes target
  class, logical/physical capacity, native gates and parameter domains, directed topology,
  result classes, execution limits, control flow, timing, pulse, noise, calibration and
  bounded auxiliary-qubit policy.
- Closed vocabularies, strict field validation and invalid-combination checks fail closed.
- Canonical semantic bytes and SHA-256 fingerprints are deterministic across hash seeds;
  non-semantic display labels do not affect identity.
- Structured comparison reports every unmet requirement without silently weakening target
  class, gates, parameter domains, topology, limits, results or advanced capabilities.
- Provider names, real backend IDs, credentials, URLs, jobs and queue state are rejected.
- Anonymous local-runtime, QASM and non-QASM fixtures cover round-trip, isolation, privacy
  and identity behavior.
- The approved private CPU budget is now an executable 10/100/1K/10K fail-closed gate.
- Stable public APIs, default execution, adapters, provider code and legacy behavior remain
  unchanged.

## Performance evidence

The approved seven-iteration, two-warmup Docker validation passed all latency, memory and
determinism thresholds. At 10,000 capability entries, construction plus fingerprinting
measured 3.226471 ms p95 against a 12 ms budget, compatible comparison measured
11.580995 ms p95 against a 50 ms budget, and peak traced host memory was 1,531,880 bytes
against a 4,194,304-byte budget.

These are private regression limits, not public service-level commitments.

## Batch B proposed boundary

Batch B may add only a private TargetIR model and deterministic QuantumIR-to-TargetIR
legalization after separate owner approval. Its required evidence is:

- typed target operations and explicit logical/physical layout semantics;
- gate, parameter-domain, result and directed-topology legality against a capability
  snapshot;
- deterministic lowering and target-dependent identity;
- unsupported operations and missing/unknown capabilities fail closed;
- structural, state, order and trainable-gradient differential evidence where applicable;
- an independently measured performance baseline before any budget proposal;
- zero public/default-path impact.

Batch B does not authorize ExecutableArtifact, emitters, RuntimeAdapter, conformance,
shadow/default execution, provider SDKs, remote submission, real backend IDs, public API
changes, legacy retirement, or Batches C-H.

## Requested decision

Approve Batch A exit and Batch B entry with:

```text
approve IR-PHASE3-BATCH-A-EXIT-BATCH-B
```
