# API Change Proposal 056: Regional Twin candidate holdout gate

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice compares an incumbent and a later regional Twin against the same future
QPU results, with a prospectively separated holdout group.

## Problem

`TwinCandidateSuite` can compare two local Twins on shared hardware results,
and `TwinRegionHoldoutStudy` can separate reference and holdout circuits for one
regional Twin. Neither object proves that a later regional candidate improves
on circuits excluded from its reference group. Applications would otherwise
have to combine topology checks, circuit separation, simultaneous confidence,
and promotion logic outside the framework.

## Decision

- Add `TwinRegionCandidateHoldoutStudy` and
  `prepare_region_candidate_holdout(...)`.
- Require incumbent and candidate regional models to describe the same target,
  physical-qubit order, directed topology, supported operations, and structural
  limits. The candidate calibration capture must be later.
- Freeze disjoint reference and holdout candidate suites before provider work.
  Every circuit must be structurally covered by both regional models.
- Reuse one authoritative QPU result per circuit to compare both frozen model
  predictions. Do not submit a separate incumbent task.
- Allocate confidence across reference and holdout groups and then across the
  circuits in each existing candidate suite.
- Add `TwinRegionCandidateHoldoutEvaluation`. A candidate is `improved` only
  when the holdout group is improved and the reference group is not degraded.
  It is `degraded` when either group is degraded; all other cases are
  `inconclusive`.
- Keep submission explicit through each existing candidate trial. The study
  never submits, polls, retries, replaces a task, promotes a model, or changes
  workload routing.
- Persist the study with a strict private create-once file. The evaluation is
  deterministic and JSON-ready through `to_dict()`; a second persistence layer
  is deferred until there is a concrete resume requirement.

## Public API

```python
study = fq.twin.prepare_region_candidate_holdout(
    incumbent_region,
    candidate_region,
    reference_circuits,
    holdout_circuits,
    physical_qubits=(20, 27, 34),
    name="shenglian-region-candidate",
    shots=1024,
)

fq.twin.dump_region_candidate_holdout_study(
    study,
    "region-candidate-holdout.json",
)

# Applications explicitly submit each trial and retain its
# TwinCandidateSubmission. After every task is terminal:
evaluation = study.validate_results(
    reference_submissions,
    reference_results,
    holdout_submissions,
    holdout_results,
    reference_circuits=reference_circuits,
    holdout_circuits=holdout_circuits,
    confidence_level=0.95,
)

print(evaluation.decision)
print(evaluation.holdout_evaluation.mean_candidate_improvement)
print(evaluation.holdout_evaluation.candidate_improvement_lower_bound)
```

## Compatibility

The functions, immutable types, and schemas are additive. Existing candidate,
regional, holdout, persistence, and submission behavior remains unchanged.

## Acceptance

- Preparation and loading perform no provider operation.
- Region target, mapping, topology, operations, limits, and circuit coverage
  must match exactly; mismatches fail closed.
- Reference and holdout circuit identities are disjoint.
- One QPU task per circuit is shared by incumbent and candidate comparisons;
  task identities are globally distinct across both groups.
- The requested confidence covers both groups and every circuit.
- The holdout group is necessary for an `improved` decision, while reference
  degradation vetoes improvement.
- Metrics concern classical measurement distributions, not state fidelity,
  arbitrary-circuit accuracy, causal training benefit, or production trust.
