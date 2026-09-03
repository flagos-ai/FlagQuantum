# Deployment Bridge Stage 6 implementation review

Updated: 2026-09-03

## Conclusion

The approved private transport-neutral connector contracts and in-memory scripted
transport are implemented. Four anonymous flows cover success, provider rejection,
cancellation and unknown submission requiring reconciliation.

The connector separates program, artifact, request, idempotency, provider namespace,
requested/actual target and credential-reference identities. Its evidence is explicitly
labelled `offline_scripted_evidence`; it cannot claim sandbox or production certification.

## Safety evidence

- Operation, lifecycle, terminal-state and error taxonomies match the proposal.
- Every operation consumes exactly one response from a finite immutable transcript.
- Missing, extra, reordered or mismatched operations fail closed.
- Target, capability, namespace, expiry and safety mismatches fail closed.
- Unknown submission performs exactly one scripted submit, preserves idempotency and
  has no retry or fallback operation.
- Credential references contain no secret, endpoint, environment name or store path
  and cannot be resolved.
- The implementation has no Runtime, provider SDK, network, socket, filesystem,
  environment, subprocess, callback or background-worker path.
- Public API, deployment, provider, Runtime, shadow and default paths are unchanged.

## Independent baseline

Seven measured samples with two warmups in `flagquantum-dev:pr-check` produced:

| Connector evaluations | p95 | Peak traced host memory |
| ---: | ---: | ---: |
| 10 | 0.665 ms | 10,707 B |
| 100 | 5.360 ms | 10,707 B |
| 1K | 52.576 ms | 10,739 B |
| 10K | 643.553 ms | 10,739 B |

This is a private CPU baseline, not a budget or public SLA. Proposed regression
envelopes are 5/25/125/1250 ms and 64/128/512/1024 KiB for 10/100/1K/10K cases.
They remain inactive until separately approved.

## Explicitly absent

There is no live sandbox transport, Runtime integration, provider SDK/network,
credential resolution, real backend/job identity, real execution/submission,
retry/fallback, shadow/canary, sampling/routing, telemetry, migration, public/default
change, product-roadmap Stage 6 work or unified IR Phase 4.

## Review decision requested

Approve only the private regression budget and machine gate with:

```text
approve DEPLOYMENT-BRIDGE-STAGE6-PERFORMANCE-BUDGET
```
