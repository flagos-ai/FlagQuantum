# API Change Proposal 058: Bounded regional Twin release

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice turns an explicitly improved regional candidate holdout evaluation into
an immutable, exact-circuit release manifest.

## Problem

FlagQuantum can freeze, validate, and persist a regional incumbent/candidate
holdout study, but applications lack one canonical artifact saying which
candidate passed, against which evidence, and for exactly which scope. Treating
an `improved` string as a global accuracy or routing authorization would exceed
the evidence.

## Decision

- Add `TwinRegionRelease` and `release_region_candidate(...)`.
- Require exact incumbent, candidate, study, suite, evaluation, target,
  snapshot, ordered mapping, topology, operations, and structural-limit
  identity agreement.
- Require an overall `improved` decision: the holdout confidence interval must
  show improvement and the reference group must not be degraded.
- Freeze both circuit groups and their confidence-qualified improvement
  intervals. The release scope is `exact_circuits`; it makes no arbitrary
  circuit claim.
- Add strict, private, create-once `dump_region_release(...)` and
  `load_region_release(...)` functions.
- A release never contacts a provider, mutates the active model, submits or
  retries a task, or authorizes workload routing. Applications retain the
  explicit selection and deployment decision.

## Public API

```python
release = fq.twin.release_region_candidate(
    incumbent_region,
    candidate_region,
    study=study,
    evaluation=evaluation,
)

fq.twin.dump_region_release(release, "region-release.json")
restored = fq.twin.load_region_release("region-release.json")

assert restored.candidate_region_identity == candidate_region.identity
assert restored.scope == "exact_circuits"
assert restored.routing_authorized is False
print(restored.verified_circuit_identities)
```

`study` and `evaluation` are the frozen and completed artifacts from the
regional candidate holdout workflow. The release call is the application's
explicit act; no background or natural-language control path exists.

## Compatibility

The type and three functions are additive. Existing Twin model, candidate,
holdout, evidence, submission, prediction, and provider behavior is unchanged.

## Acceptance

- Only a matching `improved` holdout evaluation can create a release.
- Model, study, suite, evaluation, task target, topology, mapping, and snapshot
  mismatches fail closed.
- The manifest names exact verified circuits and cannot authorize routing.
- Dump/load round trips preserve equality and deterministic identity.
- Missing, extra, modified, malformed, or noncanonical payloads fail closed.
- Files use mode `0600`, are create-once, and allow identical idempotent writes.
- No provider operation, model mutation, automatic promotion, or routing occurs.
