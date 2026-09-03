# Deployment Bridge Stage 7 entry proposal

Updated: 2026-09-03

## Purpose

Stage 7 proposes a guarded path from the Stage 6 offline scripted connector to exactly
one named Provider's non-billable live sandbox. This document does not select that
Provider and does not authorize implementation, network access, credential resolution
or submission.

Deployment Bridge Stage 7 is distinct from product-roadmap Stage 7, production
activation, multi-provider routing and public API stabilization.

## Mandatory provider selection

An owner must explicitly name one canonical Provider namespace and one documented
non-billable sandbox target. The selection must bind authoritative sandbox
documentation, discovery and backend-identity rules, capabilities, endpoints, region,
jurisdiction, retention, quotas and an accountable owner. Repository history or an
existing adapter must never be treated as implicit selection.

Provider selection is evidence, not Stage 7 Entry approval and not live-access
approval.

## Isolation boundary

Any future Provider-specific implementation remains private under
`flagquantum/_compiler/provider_sandboxes/`. It is not publicly exported or registered
by default. Provider backend and job identities remain execution evidence and never
enter program IR. An external adapter repository must not carry FlagQuantum core
source.

Network access is deny-by-default. A future live-access package must bind exact HTTPS
destinations and operations, sandbox target attestation, a least-privilege ephemeral
credential handle, and a kill switch. Provider responses and environment values cannot
expand the endpoint allowlist.

## First live probe ceiling

A future, separately approved probe is limited to one reviewed synthetic circuit, one
submission, one job, at most 100 shots, 20 polls, one cancellation, five minutes and
zero estimated cost. Retry and fallback are forbidden. An unknown submission outcome
stops all later submission and requires manual reconciliation with the same
idempotency identity.

Audit evidence excludes credentials, personal data, raw programs and raw results.

## Approval ladder

1. Current step: proposal and offline contract tests only.
2. Owner selects one Provider and one documented non-billable sandbox.
3. Independent Stage 7 Entry approval may authorize private implementation only.
4. Implementation readiness and its performance budget receive a separate review.
5. A final, expiring live-access approval binds the exact target, endpoint, credential
   scope and probe before any network access.

No step implies the next, and sandbox approval never implies production access.

## Next owner action

Stage 7 Entry cannot be approved yet. Supply an explicit selection using:

```text
approve DEPLOYMENT-BRIDGE-STAGE7-PROVIDER-SELECTION provider=<canonical-provider-namespace> sandbox=<documented-non-billable-sandbox-target>
```
