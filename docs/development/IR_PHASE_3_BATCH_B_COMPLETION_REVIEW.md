# FlagQuantum IR Phase 3 Batch B completion review

Status: implementation and approved performance gate complete; owner exit approval pending.

## Delivered

- Private immutable `TargetIR` operations carry explicit physical-qubit operands and only
  schema-valid gate parameters.
- QuantumIR-to-TargetIR legalization validates source IR, target-native gates, parameter
  domains, result classes, shots, operation limits, explicit layout and directed topology.
- Target program identity binds source program identity, capability fingerprint, layout,
  ordered operations, result requirements and requested shots.
- Dynamic bindings retain identity only when the target declares parameter-binding support;
  constrained dynamic values that cannot be proven legal fail closed.
- Unsupported inputs return structured diagnostics without decomposition, routing,
  capability weakening or a partial TargetIR result.
- Static state, operation order and trainable-gradient differential evidence passes.
- Direct TargetOperation construction rejects non-schema fields, preserving the boundary
  against credentials and provider/runtime metadata.
- The approved private CPU budget is now an executable 10/100/1K/10K fail-closed gate.
- Stable public APIs, default execution, adapters and legacy behavior remain unchanged.

## Performance evidence

The approved seven-iteration, two-warmup Docker validation passed all latency, memory and
determinism thresholds. At 10,000 operations, legalization measured 132.36428 ms p95
against a 250 ms budget, and peak traced host memory was 9,188,711 bytes against a
33,554,432-byte budget.

These are private regression limits, not public service-level commitments.

## Batch C proposed boundary

Batch C may add only a private artifact-profile registry and sealed `ExecutableArtifact`
after separate owner approval. Its required evidence is:

- closed format/version profiles with explicit media and text/binary semantics;
- immutable payload bytes and canonical payload/content hashes;
- a complete source-program, TargetIR, target-capability, compilation and artifact identity
  chain;
- profile/payload mismatch and post-seal tampering rejection;
- credential, token, provider job and dispatch-locator exclusion from portable artifacts;
- deterministic sealing for anonymous runtime-plan, QASM and non-QASM fixtures;
- an independently measured performance baseline before any budget proposal;
- zero public/default-path impact.

Batch C does not authorize new emitters, RuntimeAdapter, execution, conformance,
shadow/default integration, provider SDKs, remote submission, real backend IDs, public API
changes, legacy retirement, or Batches D-H.

## Requested decision

Approve Batch B exit and Batch C entry with:

```text
approve IR-PHASE3-BATCH-B-EXIT-BATCH-C
```
