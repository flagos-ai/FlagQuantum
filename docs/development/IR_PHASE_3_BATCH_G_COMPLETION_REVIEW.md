# FlagQuantum IR Phase 3 Batch G completion review

Updated: 2026-09-03

## Decision summary

Batch G is technically ready for an explicit owner exit decision. It delivers the
authorized deployment-compatibility and canary-readiness proposal plus offline contract
tests. It adds no runtime bridge and activates no migration stage.

The existing `DeploymentPackage` and provider path remain unchanged and authoritative.

## Completed evidence

- The inventory is pinned to the seven current `DeploymentPackage` dataclass fields and
  the current package schema constant.
- The proposal identifies the current Phase 3 artifact, binding, receipt and result
  owners without exposing them through the stable namespace.
- Program payload, source IR, shots, backend locator, semantic capabilities, routing
  evidence, display name and provider extensions each have an explicit mapping class and
  identity rule.
- The compatibility matrix keeps every current default unchanged and leaves unsupported
  Phase 3 submission, dual execution and automatic fallback inactive.
- Six migration stages are ordered and separately gated; only Stage 0 proposal and
  inventory is authorized.
- Rollback retains the legacy path and does not require rewriting user packages or stored
  receipts.
- Canary readiness requires correctness, privacy, resource, queue, cost, conformance,
  observability, incident and public compatibility evidence plus separate approval.
- Current activation blockers are recorded rather than hidden.
- Public API snapshots, `DeploymentPackage` schema, provider behavior and default paths
  remain unchanged.

## Important finding

The legacy deployment digest and Phase 3 artifact identity are intentionally not
equivalent. The legacy digest combines deployment name, target locator, shots, program
and routing evidence. Phase 3 separates portable artifact identity from execution
options and external provider/job identity. Any later bridge must recompute and verify
the Phase 3 chain; digest reuse is prohibited.

## Batch H boundary

If separately approved, Batch H may perform only the aggregate Phase 3 exit review. It
must bind the authorization chain and evidence for Batches A-G, verify the local, QASM
and non-QASM target families share the artifact/runtime boundary, verify every approved
performance gate, list residual blockers, and assess public/default impact and rollback.

Batch H does not authorize runtime integration, provider submission, canary activation,
production/default shadowing, deployment migration, public API changes, default-path
changes, legacy retirement, or Phase 4 entry.

## Requested decision

Approve Batch G exit and Batch H aggregate-review entry with:

```text
approve IR-PHASE3-BATCH-G-EXIT-BATCH-H
```

Phase 3 exit remains a later, separate owner decision.
