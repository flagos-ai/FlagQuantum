# API Change Proposal 045: Twin candidate circuit suite

## Status

Approved for implementation. On 2026-09-14, the API owner authorized the next
Twin increment after merging resumable candidate submissions.

## Problem

One prospective hardware result can compare two frozen Twins fairly for one
fixed circuit, but it cannot establish that a candidate is consistently closer
to a QPU over a useful workload sample. Reusing that one result for a broader
claim would confuse exact-circuit evidence with workload evidence.

## Additive API

```python
circuits = (
    fq.Circuit(2).h(0).cx(0, 1),
    fq.Circuit(2).x(0).cx(0, 1),
    fq.Circuit(2).h(1).cx(1, 0).x(1),
)
suite = fq.twin.prepare_candidate_suite(
    incumbent,
    candidate,
    circuits,
    name="candidate-workloads",
    shots=1024,
)
fq.twin.dump_candidate_suite(suite, "candidate-suite.json")
```

Applications explicitly submit and checkpoint each trial with the existing
single-task API:

```python
submissions = []
for index, trial in enumerate(suite.trials, start=1):
    receipt = trial.experiment.submit(provider)  # one explicit QPU task
    submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
    fq.twin.dump_candidate_submission(
        submission,
        f"candidate-submission-{index:02d}.json",
    )
    submissions.append(submission)
```

After fetching the corresponding terminal results:

```python
evaluation = suite.validate_results(
    submissions,
    results,
    circuits=circuits,
    confidence_level=0.95,
)

print(evaluation.decision)
print(f"incumbent: {evaluation.mean_incumbent_qpu_agreement:.2%}")
print(f"candidate: {evaluation.mean_candidate_qpu_agreement:.2%}")
print(f"ideal SV: {evaluation.mean_ideal_qpu_agreement:.2%}")
```

The public additions are `TwinCandidateSuite`,
`TwinCandidateSuiteEvaluation`, `prepare_candidate_suite()`,
`dump_candidate_suite()`, and `load_candidate_suite()`.

## Statistical semantics

Every circuit has its own prospectively frozen incumbent and candidate
prediction, exact submitted OpenQASM, physical mapping, receipt, and hardware
counts. Per-circuit confidence is Bonferroni-adjusted so all reported intervals
hold simultaneously at the requested suite confidence. The suite decision uses
the mean candidate improvement and the mean of the simultaneously valid
per-circuit error radii. `improved` or `degraded` requires that interval to
exclude zero; otherwise the result is `inconclusive`.

The result describes mean agreement for this predeclared circuit suite on one
ordered physical mapping. It is not amplitude fidelity, arbitrary-circuit
accuracy, causal attribution, or permission to promote a model.

## Submission and failure boundary

Suite preparation, persistence, and evaluation are offline. The Suite has no
bulk-submit method. Every task is created only by an explicit existing
`trial.experiment.submit(provider)` call and should be checkpointed before the
next task is submitted. Missing, reordered, duplicate, or mismatched trials,
receipts, results, circuits, snapshots, targets, mappings, QASM, or task IDs fail
closed.

FlagQuantum does not collect calibration automatically, poll implicitly,
retry, schedule, select workloads, promote a Twin, route user jobs, or expose
Agent/MCP behavior. The complete Quafu example separates `prepare`, indexed
single-task `submit`, and submission-free `evaluate` commands.

## Compatibility

The change is additive. Existing Twin v1 types, signatures, schemas, candidate
decision meanings, and provider ownership remain unchanged.
