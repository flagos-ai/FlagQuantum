# FlagQuantum IR Phase 3 Batch F completion review

Updated: 2026-09-03

## Decision summary

Batch F is technically ready for an explicit owner exit decision. Its authorized scope
is complete: a private, explicitly constructed shadow harness compares a candidate path
without changing the authoritative legacy result. The approved performance budget has
also passed its machine gate.

This review does not activate shadowing in production or default execution and does not
authorize Batch G by itself.

## Completed evidence

- Legacy executes first and its exact observation remains authoritative.
- The candidate result is not exposed through the outcome contract.
- Status, type, shape, value, identity, candidate-failure and limit-breach outcomes form
  a closed mismatch taxonomy.
- Evidence contains bounded sizes, timing and hashes only; raw inputs, results and
  exception text are absent.
- Count, input-size, evidence-size, candidate-time and mismatch breakers fail closed.
- The one-way kill switch and comparison reservation are thread-safe.
- Anonymous evidence identities remain stable across Python hash seeds.
- The harness has no environment, import-hook, global, background, telemetry, provider,
  network, public API or default-path activation.
- The approved 10/100/1K/10K private CPU budget passes. At 10K comparisons, observed
  p95 was 228.797917 ms against a 600 ms ceiling and peak traced memory was 5,751 bytes
  against a 1,048,576-byte ceiling.

## Batch G boundary

If separately approved, Batch G may prepare only a private deployment-compatibility and
canary-readiness proposal. It must specify deterministic mapping between the existing
`DeploymentPackage` boundary and the verified Phase 3 artifact/runtime boundary,
compatibility checks, rollback criteria, migration stages and public/default-path impact.

Batch G must not activate a canary, attach shadowing to `fq.run` or `fq.plan`, submit to a
provider, call a network or SDK, change a public API, switch a default path, migrate user
state, retire legacy execution or claim production readiness.

## Requested decision

Approve Batch F exit and Batch G proposal-only entry with:

```text
approve IR-PHASE3-BATCH-F-EXIT-BATCH-G
```

Batch G implementation beyond an offline compatibility proposal, canary activation,
Batch H and Phase 3 exit remain separately gated.
