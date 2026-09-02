# FlagQuantum IR Phase 3 Batch B implementation review

Updated: 2026-09-02

## Result

Batch B implements a private `TargetIR` and deterministic, provider-free
QuantumIR-to-TargetIR legalization boundary. It remains absent from the stable namespace
and unreachable from `fq.run`, `fq.plan`, deployment and adapter paths.

The implementation provides:

- immutable typed target operations with explicit physical-qubit operands;
- explicit logical-to-physical layout and required-result semantics;
- identity bound to source program, target capability snapshot, layout, operation order,
  required results and requested shots;
- gate, arity, type, parameter-domain, result, shot, program-size and directed-topology
  legality checks;
- dynamic parameter preservation only for targets declaring binding support;
- fail-closed handling when a dynamic value cannot be proven inside a constrained domain;
- no implicit gate decomposition, routing, capability weakening or partial TargetIR result;
- state, order and trainable-gradient differential evidence for the identity lowering.

Provider identity, real backend IDs, credentials, dispatch locations and runtime lifecycle
remain outside TargetIR and compilation identity.

## Evidence

- Focused authorization, model, legality, failure, identity, state, order, gradient,
  privacy and namespace tests pass.
- Target program identity is stable across Python hash seeds.
- The 10/100/1K/10K operation baseline is recorded in
  `contracts/ir-phase3-batch-b-performance-baseline.json`.
- At 10K operations, observed legalization p95 was 98.144071 ms and peak traced host
  memory was 9,188,711 bytes.

The measurements are environment-specific observations, not a public SLA. Proposed
private regression envelopes remain unapproved in
`tests/fixtures/internal_ir/phase3_batch_b_performance_budget_candidate.json`.

## Remaining closed surfaces

Batch B does not implement or authorize ExecutableArtifact, emitters, RuntimeAdapter,
conformance, shadow/default execution, provider integration, real backend IDs, public API
changes, legacy retirement, or Batches C-H.

## Review decision requested

Approve only the private Batch B performance budget and its machine regression gate with:

```text
approve IR-PHASE3-BATCH-B-PERFORMANCE-BUDGET
```

Batch B exit and Batch C remain separately gated.
