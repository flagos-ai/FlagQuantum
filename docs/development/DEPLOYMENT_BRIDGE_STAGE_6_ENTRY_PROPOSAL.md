# Deployment Bridge Stage 6 entry proposal

Updated: 2026-09-03

## Purpose

Stage 6 proposes a private transport-neutral connector contract for a future controlled
Provider sandbox. Its first implementation remains offline: it can execute only against
a finite immutable scripted transcript and cannot open a network connection.

This Deployment Bridge stage is distinct from product-roadmap Stage 6 hybrid
orchestration, a production connector and live-access approval.

## Contract boundary

The connector separates program, artifact, request, idempotency, provider namespace,
requested target, actual target and credential-reference identities. A credential
reference contains only anonymous identity, scope and expiry; it contains no secret,
endpoint, environment-variable name or store path and cannot be resolved.

The normalized lifecycle is:

```text
prepared -> admitted -> submission_started -> accepted -> running
                                               |          |
                                               |          +-> succeeded / failed
                                               +-> cancelled
                                               +-> outcome_unknown -> reconciliation
```

Only the scripted transcript can model these facts. Unknown submission never retries
or falls back, keeps the same idempotency identity and blocks later submission until
reconciled. Terminal states are irreversible.

## Two-step activation

1. This entry may later authorize private connector contracts, scripted transport,
   anonymous fixtures and an independent CPU baseline.
2. A separate future approval would be required for one named live sandbox transport
   and credential-reference resolution, supported by external readiness evidence.

Step 1 does not imply Step 2. Neither step permits production targets or billable work.

## Entry decision requested

```text
approve DEPLOYMENT-BRIDGE-STAGE6-ENTRY
```
