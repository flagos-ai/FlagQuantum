# API Change Proposal 053: Regional Twin holdout evaluation persistence

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice persists the result of an already completed regional holdout validation.

## Problem

`TwinRegionHoldoutStudy` can be frozen and restored before hardware work, but
the resulting `TwinRegionHoldoutEvaluation` can only be held in memory. A
dashboard, audit, or later offline analysis therefore cannot consume the exact
validated result without application-specific serialization.

## Decision

- Add `dump_region_holdout_evaluation(...)` and
  `load_region_holdout_evaluation(...)`.
- Preserve the existing `flagquantum.twin_region_holdout_evaluation.v1`
  payload exactly; no schema field or meaning changes.
- Restore every nested `TwinRegionSuiteEvaluation` and
  `TwinValidationSeries` under their existing strict version-1 schemas.
- Recompute all derived counts, agreements, repeatability metrics, uncertainty,
  and TV bounds. Reject changed, missing, extra, or non-canonical values.
- Write private mode-`0600`, create-once checkpoints. Rewriting identical
  content is idempotent; different existing content is never replaced.
- Perform no provider access, task submission, polling, retry, model update,
  trust promotion, or routing decision.

## Public API

```python
evaluation = study.validate_results(
    reference_submissions,
    reference_results,
    holdout_submissions,
    holdout_results,
    reference_circuits=reference_circuits,
    holdout_circuits=holdout_circuits,
    confidence_level=0.95,
)

fq.twin.dump_region_holdout_evaluation(
    evaluation,
    "region-holdout-evaluation.json",
)
restored = fq.twin.load_region_holdout_evaluation(
    "region-holdout-evaluation.json",
)

print(restored.reference_twin_qpu_agreement)
print(restored.holdout_twin_qpu_agreement)
print(restored.holdout_twin_qpu_tv_increase)
print(restored.holdout_simultaneous_tv_error_bound)
```

The complete checkpointed Quafu workflow remains
`examples/remote/quafu_twin_region_holdout.py`; its `evaluate` command now
writes the evaluation artifact in addition to exact-circuit support artifacts.

## Compatibility

The two functions are additive. Existing classes, signatures, schemas,
submission behavior, evaluation behavior, and serialized artifacts remain
unchanged.

## Acceptance

- Dump/load round trips preserve equality and identity.
- Every nested suite and validation series is reconstructed and validated.
- Changed derived metrics, malformed nested records, missing or extra fields,
  and non-canonical payloads fail closed.
- Files are private and create-once, with identical-content idempotence.
- The documented example is complete and executable.
- Persistence performs no provider operation.
