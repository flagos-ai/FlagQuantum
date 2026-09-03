# Deployment Bridge Stage 3 entry proposal

Updated: 2026-09-03

Status: **proposal only — no canary implementation or activation**

## Conclusion

Stage 2 proves that an eligible legacy package can traverse the new compilation and
artifact chain offline. It does not prove that FlagQuantum can safely place real work on
a second execution path. Stage 3 should therefore build only a private, deterministic
**canary-readiness evaluator**: it answers whether all evidence needed for a later,
separately approved activation review is present. It cannot activate or execute anything.

This Deployment Bridge stage is distinct from the completed unified IR Phase 3
provider-conformance work. The existing conformance harness is provider-neutral and uses
local/synthetic adapters; it does not certify a real provider integration.

## Proposed boundary

The proposed private operation is
`evaluate_deployment_canary_readiness(dry_run_report, conformance, controls, policy)`.
It consumes immutable, anonymous evidence snapshots and returns an immutable closed
readiness report. It must reject Runtime adapters, executable artifacts, credentials,
provider SDK clients, endpoints, real backend/job identifiers and raw programs/results.

Its strongest possible result is `ready_for_separate_activation_review`, not “enabled,”
“active,” or “safe to submit.” Missing, stale, revoked, mismatched or ambiguous evidence
always yields `blocked`.

## Safety contract

A future canary must require explicit per-request opt-in. Global configuration or an
environment variable cannot opt users in, and the existing `DeploymentPackage` path
remains authoritative until a later routing decision is separately approved.

Before any future submission, the system must have a dispatch-attempt identity and an
at-most-once rule. An unknown submission outcome must never trigger automatic retry or
fallback submission: doing so can duplicate QPU work, consume quota twice, create
conflicting receipts and charge the user twice.

The kill switch must be fail-closed, evaluated before sampling/dispatch, independently
owned and auditable. Rollback must restore the unchanged legacy path without rewriting
packages or receipts. Correctness, privacy, latency, memory, sampling, concurrency,
queue, quota and monetary-cost budgets all require independent evidence and approval.

## Identity, privacy and operations

Portable readiness evidence may contain semantic hashes, closed findings and budget
booleans. Real provider/backend/job/account/user identifiers, credentials, endpoints,
raw programs, results, metadata and diagnostics are forbidden. If operational locators
are ever approved, they belong in a separate restricted store with explicit access,
retention and verified-deletion controls.

Activation also requires named owners for approval, kill switch, provider adapter,
privacy/retention, incident response and cost/quota, together with recovery objectives
and an offline failure-injection exercise.

## Rollout ladder

The stages are separately gated and do not inherit permission:

1. Private offline readiness evaluator.
2. Offline failure injection and operator rehearsal.
3. Provider-specific sandbox conformance.
4. Production shadow or dual execution.
5. Bounded candidate-authoritative routing.
6. Public/default migration.

The next approval can authorize only step 1, anonymous fixtures and an independent
performance baseline. It cannot authorize execution, submission, sampling, routing,
Runtime access, a real provider, telemetry or public/default integration.

## Entry decision requested

Authorize only the private offline evaluator with:

```text
approve DEPLOYMENT-BRIDGE-STAGE3-ENTRY
```
