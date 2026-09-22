# API Change Proposal 052: Regional Twin holdout validation

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice prospectively separates reference circuits from circuits held out of the
reference validation group.

## Problem

Agreement on circuits already selected as the regional reference suite does
not establish that a Twin predicts different circuits. Independent suites can
be run today, but they do not preserve a predeclared reference/holdout split,
prevent circuit or task leakage across groups, or provide one simultaneous
confidence statement.

## Decision

- Add `prepare_region_holdout_study(...)` and immutable
  `TwinRegionHoldoutStudy`.
- Freeze two disjoint regional validation suites against one regional Twin,
  mapping, shot count, and repetition count before provider work begins.
- Persist the complete study with strict, private, create-once semantics so the
  split can be audited after execution.
- Keep task submission explicit through the existing `TwinExperiment` and
  `TwinSubmission` APIs. The study never submits, polls, retries, or replaces a
  task.
- Validate all reference and holdout tasks together. Task identities must be
  globally distinct across both groups.
- Allocate confidence first across the two groups and then through each
  existing suite and repetition correction.
- Add immutable `TwinRegionHoldoutEvaluation` with separate reference and
  holdout agreement, ideal-baseline, repeatability, uncertainty, and exact
  support records.
- Report `holdout_twin_qpu_tv_increase` as holdout mean Twin-QPU TV error minus
  reference mean Twin-QPU TV error. Positive values mean the holdout circuits
  were harder for the Twin.
- Do not call this arbitrary-circuit accuracy or a model-training result. It is
  evidence only for the prospectively declared holdout circuits.

## Public API

```python
study = fq.twin.prepare_region_holdout_study(
    region_twin,
    reference_circuits,
    holdout_circuits,
    physical_qubits=(20, 27, 34),
    name="shenglian-region-holdout",
    shots=1024,
    repetitions=2,
)

print(study.reference_circuit_count)
print(study.holdout_circuit_count)
print(study.planned_task_count)
print(study.planned_shots)

fq.twin.dump_region_holdout_study(study, "region-holdout-study.json")
study = fq.twin.load_region_holdout_study("region-holdout-study.json")

# Applications submit every frozen experiment explicitly and retain each
# TwinSubmission. After all tasks are terminal:
evaluation = study.validate_results(
    reference_submissions,
    reference_results,
    holdout_submissions,
    holdout_results,
    reference_circuits=reference_circuits,
    holdout_circuits=holdout_circuits,
    confidence_level=0.95,
)

print(evaluation.reference_twin_qpu_agreement)
print(evaluation.holdout_twin_qpu_agreement)
print(evaluation.holdout_twin_qpu_tv_increase)
print(evaluation.holdout_simultaneous_tv_error_bound)
```

The complete checkpointed Quafu workflow is
`examples/remote/quafu_twin_region_holdout.py`.

## Compatibility

The new functions and types are additive. Existing regional models, suites,
evaluations, histories, submission behavior, and serialized schemas remain
unchanged.

## Acceptance

- Preparing and loading a study performs no provider operation.
- Reference and holdout circuit identities are disjoint before the study is
  returned.
- Both groups use one regional Twin identity, mapping, shot count, and
  repetition count.
- Task identities are distinct across both groups and incomplete or reordered
  task collections fail closed.
- The requested confidence covers every circuit and repetition in both groups.
- Metrics describe classical measurement distributions, not state fidelity,
  arbitrary-circuit accuracy, training generalization, or production trust.
