# API Change Proposal 054: Regional Twin holdout history

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice aligns already completed prospective holdout evaluations across
calibration states.

## Problem

`TwinRegionHoldoutEvaluation` distinguishes fixed reference circuits from
prospectively held-out circuits at one calibration snapshot. It cannot express
whether that same frozen experimental design remains comparable over later
calibration states without application-specific alignment and serialization.

## Decision

- Add `TwinRegionHoldoutHistory`, `build_region_holdout_history(...)`,
  `dump_region_holdout_history(...)`, and `load_region_holdout_history(...)`.
- Require one provider/backend, ordered physical mapping, full directed
  topology, ordered reference suite, ordered holdout suite, and experimental
  design across all observations.
- Require strictly increasing capture times and unique snapshot, regional-model,
  study, and hardware-report identities.
- Recompute all derived agreement, repeatability, uncertainty, count, and shot
  arrays while loading. Reject changed, missing, extra, or non-canonical data.
- Write private mode-`0600`, create-once artifacts. Identical content is
  idempotent; different existing content is never replaced.
- Perform no provider access, submission, polling, retry, model update,
  trust-window inference, promotion, or routing decision.

## Public API

```python
import flagquantum as fq

state_01_twin = build_region_twin_for_state_01()
state_02_twin = build_region_twin_for_state_02()

state_01_evaluation = fq.twin.load_region_holdout_evaluation(
    "state-01-holdout-evaluation.json"
)
state_02_evaluation = fq.twin.load_region_holdout_evaluation(
    "state-02-holdout-evaluation.json"
)

history = fq.twin.build_region_holdout_history(
    [
        (state_01_twin, state_01_evaluation),
        (state_02_twin, state_02_evaluation),
    ]
)

print(history.reference_twin_qpu_agreements)
print(history.holdout_twin_qpu_agreements)
print(history.holdout_tv_error_increases)
print(history.holdout_simultaneous_tv_error_bounds)

fq.twin.dump_region_holdout_history(history, "region-holdout-history.json")
restored = fq.twin.load_region_holdout_history("region-holdout-history.json")
```

`build_region_twin_for_state_01()` and `build_region_twin_for_state_02()` stand
for the same public `fq.twin.compose_region_twin(...)` construction already
documented for each calibration snapshot. The history itself is entirely
offline.

## Compatibility

The four names and the new serialized schema are additive. Existing classes,
signatures, schemas, submission behavior, evaluation behavior, and persisted
artifacts remain unchanged.

## Acceptance

- Comparable observations build and append immutably.
- Changed target, mapping, topology, circuit order, experiment design, capture
  order, or reused hardware reports fail closed.
- Dump/load round trips preserve equality and reject tampered derived values.
- Files are private and create-once, with identical-content idempotence.
- Documentation provides the complete offline usage sequence and limitations.
