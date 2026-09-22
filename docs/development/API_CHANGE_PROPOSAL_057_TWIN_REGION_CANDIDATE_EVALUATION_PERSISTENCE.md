# API Change Proposal 057: Regional Twin candidate evaluation persistence

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice persists the result of an already completed regional candidate holdout
evaluation.

## Problem

`TwinRegionCandidateHoldoutStudy` can be persisted before QPU work, but its
evaluation currently exists only in memory. A later audit cannot reconstruct
the exact incumbent/candidate comparison, shared hardware results, simultaneous
confidence allocation, or final holdout-gate decision without
application-specific serialization.

## Decision

- Add `dump_region_candidate_holdout_evaluation(...)` and
  `load_region_candidate_holdout_evaluation(...)`.
- Preserve the existing
  `flagquantum.twin_region_candidate_holdout_evaluation.v1` payload exactly.
- Restore every nested candidate-suite evaluation, candidate evaluation,
  hardware report, and incumbent/candidate validation record under its existing
  schema.
- Recompute all derived task counts, shot counts, TV distances, improvement
  intervals, confidence allocations, group decisions, and the final holdout
  decision. Reject changed, missing, extra, or non-canonical values.
- Write private mode-`0600`, create-once checkpoints. Rewriting identical
  content is idempotent; different existing content is never replaced.
- Perform no provider access, task submission, polling, retry, model update,
  candidate promotion, or workload routing.

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

fq.twin.dump_region_candidate_holdout_evaluation(
    evaluation,
    "region-candidate-evaluation.json",
)
restored = fq.twin.load_region_candidate_holdout_evaluation(
    "region-candidate-evaluation.json",
)

print(restored.decision)
print(restored.holdout_evaluation.mean_candidate_improvement)
print(restored.holdout_evaluation.candidate_improvement_lower_bound)
```

## Compatibility

The two functions are additive. Existing class fields, signatures, schemas,
submission behavior, evaluation behavior, and artifacts remain unchanged.

## Acceptance

- Dump/load round trips preserve equality and identity.
- Every nested record is reconstructed and validated.
- Changed derived metrics, malformed counts, malformed metadata, missing or
  extra fields, and non-canonical payloads fail closed.
- Files are private and create-once, with identical-content idempotence.
- The documented example is complete and executable after the explicitly
  submitted QPU tasks have reached terminal state.
- Persistence performs no provider operation and cannot promote a model.
