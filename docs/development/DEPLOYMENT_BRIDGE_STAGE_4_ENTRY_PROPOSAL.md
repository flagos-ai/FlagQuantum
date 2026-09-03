# Deployment Bridge Stage 4 entry proposal

Updated: 2026-09-03

Status: **proposal only — no rehearsal implementation or real action**

## Conclusion

Stage 3 can determine whether anonymous prerequisite evidence is complete, but a checklist
alone does not prove that failure procedures are coherent. Stage 4 should therefore add
only a private offline rehearsal engine. It will inject one synthetic scenario into an
immutable state machine and produce evidence of the decisions operators should take.
It cannot perform those decisions.

This Deployment Bridge stage is unrelated to unified IR Phase 4. It is also earlier than
provider sandboxing, shadow execution and production canary activation.

## Closed rehearsal model

The proposal defines thirteen scenarios covering kill-switch failure, stale/revoked
conformance, target/evidence mismatch, queue/quota/cost/privacy breaches, unknown
submission, acknowledgement/rollback timeout and retention/deletion failure.

Each scenario has an exact set of anonymous roles and simulated actions. The normal
state sequence is:

```text
not_started → injected → detected → admission_blocked →
operator_acknowledged → rollback_verified → evidence_sealed → completed
```

Missing, reordered, duplicated or unknown transitions yield `rehearsal_failed`. Even a
passing result means only `offline_rehearsal_passed`; there is deliberately no
production-ready or activation state.

## Unknown-submission rule

An unknown submission may already represent accepted QPU work. The only safe simulated
response is to block new candidate admission, freeze candidate retry, preserve the
legacy authority for future work, require reconciliation and escalate the relevant
anonymous roles. Automatic retry or fallback submission is forbidden because it could
duplicate QPU usage, receipts, quota consumption and charges.

## Isolation and privacy

The proposed engine accepts no Runtime adapter, provider client, execution object,
controller, callback, credential, endpoint, real identifier or raw program/result. All
actions are enum values inside an immutable report. It cannot use wall-clock sleeping,
workers, global queues, environment activation, filesystem or network access.

Roles are closed functional labels, never names or account identifiers. Detection,
acknowledgement, rollback and deletion durations are caller-supplied synthetic values,
not measurements of people or production systems. They are evaluated against explicit
immutable policy limits.

## Entry decision requested

The next approval may authorize only the private offline engine, anonymous fixtures,
internal tests and an independent performance baseline:

```text
approve DEPLOYMENT-BRIDGE-STAGE4-ENTRY
```

It does not authorize provider sandboxing, Runtime integration, execution, submission,
shadowing, canary activation, sampling, routing, telemetry or public/default changes.
