# Deployment Bridge Stage 5 entry proposal

Updated: 2026-09-03

## Purpose

Stage 5 proposes a private, offline evaluator for facts that would surround one
provider-sandbox observation. It does not contact a sandbox. A caller supplies
immutable anonymous facts, and the evaluator checks identity separation, lifecycle
ordering, idempotency, unknown-submission reconciliation, safety evidence and explicit
queue/quota/cost/retention limits.

This Deployment Bridge label is distinct from product-roadmap Stage 5, which concerns
the classic-compute-cloud control plane. It is also distinct from a live QPU integration
or production canary.

## Boundary

The proposed evaluator accepts no Runtime adapter, provider SDK, transport, endpoint,
credential, real backend/job identifier, executable artifact or raw program/result.
It cannot submit, retry, fall back, route, execute, activate, mutate state or export
telemetry. All fixtures remain anonymous and synthetic.

Lifecycle facts remain separate:

```text
intent -> attempt -> provider acceptance -> terminal result
                    \
                     -> unknown outcome -> reconciliation required
```

An unknown outcome preserves the same idempotency identity and forbids automatic retry
or fallback. It never assumes provider acceptance, quota release or cost refund.

## Proposed entry implementation

If separately approved, the next batch may add only:

- a private offline observation evaluator;
- immutable anonymous snapshot, policy and report contracts;
- golden success, rejection, incomplete and reconciliation fixtures;
- negative tests for reordered lifecycle facts, identity mismatch, stale safety
  evidence, limit breaches and forbidden data;
- an independent private CPU performance baseline.

It may not add a live sandbox integration, Runtime access, provider SDK/network,
credentials, real identities, execution/submission, retry/fallback, shadow/canary,
sampling/routing, telemetry, deployment migration, public/default changes, legacy
retirement or unified IR Phase 4.

## Entry decision requested

```text
approve DEPLOYMENT-BRIDGE-STAGE5-ENTRY
```
