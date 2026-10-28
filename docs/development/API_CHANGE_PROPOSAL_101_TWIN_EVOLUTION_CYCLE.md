# API Change Proposal 101: Human-gated Twin evolution cycles

## Status

**Implementation authorized; additive to the Twin framework API.** This slice
records one prospective refresh workflow from freshness assessment through
holdout evaluation. It is an append-only coordination artifact, not an
autonomous controller.

## Problem

The framework already exposes each evidence object needed to refresh a Twin,
but applications must remember which freshness result, probe plan, estimate,
candidate, and holdout evaluation belong together. Copying full evidence into
an orchestration record makes identity ambiguous and recovery expensive.
Leaving the sequence implicit also makes it too easy to skip a prospective
gate or accidentally repeat hardware work after a process restart.

## Decision

- Add `fq.twin.prepare_evolution_cycle(...)` and immutable
  `TwinEvolutionCycle` states.
- Permit only the sequence `prepared` → `parameter_estimated` →
  `candidate_fitted` → `holdout_evaluated`.
- Reference authoritative inputs by deterministic identity; do not embed
  calibration histories, hardware reports, predictions, or model payloads.
- Persist any intermediate state with create-once
  `dump_evolution_cycle` / `load_evolution_cycle` APIs.
- Produce one of three recommendations: `retain_incumbent`,
  `collect_more_evidence`, or `release_candidate`.
- Keep release explicit and separate. A `release_candidate` recommendation
  does not release, route to, or mutate any Twin.

## Public API

The complete offline executable is
[`examples/twin_evolution_cycle.py`](../../examples/twin_evolution_cycle.py):

```bash
python examples/twin_evolution_cycle.py
```

The application-level sequence is:

```python
cycle = fq.twin.prepare_evolution_cycle(
    incumbent=incumbent,
    freshness=freshness,
    probe_plan=plan,
)
cycle = cycle.with_parameter_estimate(estimate)
fq.twin.dump_evolution_cycle(cycle, "twin-evolution-estimated.json")

# A later process resumes without repeating probe collection.
cycle = fq.twin.load_evolution_cycle("twin-evolution-estimated.json")
cycle = cycle.with_candidate(candidate)
cycle = cycle.with_holdout_evaluation(evaluation)

print(cycle.stage)
print(cycle.recommended_action)
```

## Binding and identity

Preparation binds the incumbent snapshot, freshness assessment, probe plan,
canonical `provider:backend` target, and ordered physical-qubit mapping.
Subsequent transitions require the exact estimate, probe bank, candidate fit,
candidate snapshot, and holdout suite identities implied by the preceding
state. Changed targets, mappings, snapshots, or out-of-order transitions fail
closed.

The freshness and holdout-evaluation identities are canonical hashes of their
complete source payloads. The cycle stores only those hashes and the minimal
decision facts needed to validate its state. This keeps the record smaller
than the authoritative evidence it references.

## Recommendations

- A current incumbent starts with `retain_incumbent`.
- A due or evidence-insufficient incumbent starts with
  `collect_more_evidence`.
- An improved holdout produces `release_candidate`.
- A degraded holdout produces `retain_incumbent`.
- An inconclusive holdout produces `collect_more_evidence`.

These are typed recommendations, not actions. Existing explicit regional
release functions remain the only release boundary.

## Acceptance

- Legal transitions are explicit, immutable, and tested.
- Every source artifact is referenced by an authoritative identity.
- Any saved intermediate state resumes without provider submission.
- Persistence is private, create-once, canonical, and tamper rejecting.
- The cycle has no submission, polling, retry, routing, notification,
  promotion, or release method.
- The serialized record is smaller than the source artifacts it references.

## Non-goals

- No hardware submission, polling, retry, or cancellation.
- No provider credentials, Agent, MCP, natural-language, REST, or UI layer.
- No automatic Twin mutation, candidate release, workload routing, or hardware
  control.
- No replacement for the authoritative source artifacts.
