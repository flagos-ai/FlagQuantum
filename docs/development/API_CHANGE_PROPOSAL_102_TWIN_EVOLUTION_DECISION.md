# API Change Proposal 102: Explicit Twin evolution decisions

## Status

**Implementation authorized; additive to the Twin framework API.** This slice
records an application-authorized decision after reviewing one immutable Twin
evolution cycle. It neither authenticates the authority nor applies the
decision.

## Problem

`TwinEvolutionCycle` preserves the evidence sequence and returns a typed
recommendation, but intentionally has no release authority. Without a distinct
decision artifact, an application cannot audit whether an operator followed or
overrode that recommendation, when the decision was made, or which explicit
regional release was selected.

The recommendation must not be silently reinterpreted as an action. Equally,
FlagQuantum must not absorb application authentication, active-model state,
routing, notification, or provider orchestration.

## Decision

- Add immutable `TwinEvolutionDecision` records.
- Add `record_evolution_decision(...)` as the only constructor from an existing
  cycle.
- Preserve the cycle recommendation and operator decision as separate fields.
- Permit conservative `retain_incumbent` and `collect_more_evidence` decisions
  without a release.
- Permit `release_candidate` only after a completed improving holdout gate and
  only when an existing `TwinRegionRelease` matches the candidate snapshot,
  target, and ordered physical mapping.
- Persist records with create-once `dump_evolution_decision(...)` and strict
  `load_evolution_decision(...)` APIs.
- Keep the record non-executing: it does not create a release, activate a model,
  authorize routing, authenticate a user, or contact a provider.

## Public API

```python
decision = fq.twin.record_evolution_decision(
    cycle,
    decision="retain_incumbent",
    decided_at="2026-08-15T11:00:00+08:00",
    decision_authority="calibration-operator",
)
fq.twin.dump_evolution_decision(decision, "twin-evolution-decision.json")

restored = fq.twin.load_evolution_decision("twin-evolution-decision.json")
print(restored.recommended_action)
print(restored.decision)
print(restored.followed_recommendation)
```

The release path is deliberately more restrictive:

```python
decision = fq.twin.record_evolution_decision(
    cycle,
    decision="release_candidate",
    decided_at="2026-08-15T11:00:00+08:00",
    decision_authority="calibration-review-board",
    release=release,
)
```

`release` must already exist through the explicit regional release workflow.
The decision records its identity and candidate region identity; it does not
invoke that workflow.

## Identity and failure behavior

The canonical identity covers the cycle identity, recommendation, decision,
timezone-aware decision timestamp, opaque application authority, target,
ordered mapping, incumbent and candidate snapshots, holdout evaluation, and
optional release identities.

Candidate release fails closed when the cycle is incomplete, the evidence gate
does not recommend release, the release is absent, or its target, mapping, or
candidate snapshot differs. Non-release decisions reject release artifacts.
Malformed or unknown serialized fields are rejected.

## Acceptance

- Recommendation and operator decision remain independently observable.
- Release cannot cross a closed or incomplete evidence gate.
- A release decision binds an explicit matching release.
- Retain and collect decisions cannot imply a release.
- Persistence is canonical, private, create-once, and tamper rejecting.
- The complete offline example is directly executable.
- Public signatures, fields, schema, literals, and properties are contract
  tested.

## Non-goals

- No authentication, authorization service, user directory, signature, Agent,
  MCP, REST, CLI, or UI layer.
- No provider submission, polling, retry, cancellation, or calibration fetch.
- No release creation, active-model registry, automatic promotion, workload
  routing, notification, or hardware control.
