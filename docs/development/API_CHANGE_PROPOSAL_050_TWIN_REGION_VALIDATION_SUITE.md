# API Change Proposal 050: Regional Twin validation suites

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice follows exact-circuit regional validation with a fixed, repeated suite of
distinct regional circuits and one simultaneous confidence statement.

## Problem

One prospectively validated regional circuit proves only that exact circuit.
It does not show whether different interactions or circuit structures within a
connected region agree with the QPU. Running several independent validations
without correcting confidence across circuits would also make the combined
bound overconfident.

## Decision

- Add `prepare_region_validation_suite(...)` and immutable
  `TwinRegionValidationSuite`.
- Freeze at least two distinct circuits, one canonical regional mapping, one
  composed Twin snapshot, names, shots, and at least two repetitions per
  circuit before any provider operation.
- Expose the exact planned task and shot counts before submission.
- Keep submission explicit and resumable: applications submit each frozen
  `TwinExperiment` and persist an existing `TwinSubmission`. The suite never
  submits, polls, retries, or replaces a task.
- Validate all planned submissions and results together in deterministic
  circuit-major, repetition-minor order. Missing, extra, reordered, duplicate,
  or identity-mismatched tasks fail closed.
- Apply a Bonferroni confidence allocation across circuits; each existing
  `TwinValidationSeries` applies the second allocation across repetitions.
- Add immutable `TwinRegionSuiteEvaluation` with mean Twin-QPU agreement,
  Ideal-SV-QPU agreement, QPU repeatability, the simultaneous finite-shot
  radius, and the maximum simultaneous TV error bound.
- Convert the evaluation to existing `TwinCircuitSupport`, verifying only the
  exact suite circuit identities and recording only exercised directed
  couplers. No estimated bound is created for other circuits.
- Persist the frozen suite with create-once private-file semantics. Existing
  submissions, series, evidence, and support schemas remain unchanged.

## Public API

```python
suite = fq.twin.prepare_region_validation_suite(
    region_twin,
    circuits,
    physical_qubits=(20, 27, 34),
    name="shenglian-region-suite",
    shots=1024,
    repetitions=2,
)

print(suite.planned_task_count)
print(suite.planned_shots)

submissions = []
for experiment in suite.experiments:
    for _ in range(suite.repetitions):
        receipt = experiment.submit(provider)
        submissions.append(fq.twin.TwinSubmission.from_receipt(experiment, receipt))

results = [provider.fetch_result(item.receipt) for item in submissions]
evaluation = suite.validate_results(
    submissions,
    results,
    circuits=circuits,
    confidence_level=0.95,
)
support = evaluation.to_circuit_support()

print(evaluation.mean_twin_qpu_agreement)
print(evaluation.mean_ideal_qpu_agreement)
print(evaluation.mean_qpu_repeatability)
print(evaluation.simultaneous_tv_error_bound)
```

The complete checkpointed Quafu workflow is
`examples/remote/quafu_twin_region_validation_suite.py`.

## Compatibility

The new functions and types are additive. The suite composes existing
`TwinExperiment`, `TwinSubmission`, `TwinValidationSeries`,
`TwinEvidenceEnvelope`, and `TwinCircuitSupport` behavior without changing
their signatures or serialized meanings.

## Acceptance

- Preparing a suite performs no provider I/O and exposes exact planned tasks
  and shots.
- Every circuit passes regional mapping and topology gates before the suite is
  returned.
- At least two distinct circuits and two repetitions are required.
- All tasks are distinct and identity-bound; a partial or reordered collection
  fails closed.
- The suite confidence covers all circuits and repetitions simultaneously.
- Result support verifies only exact suite circuits and exercised couplers;
  arbitrary regional circuits remain unverified.
- Metrics describe classical measurement distributions, not state fidelity.
