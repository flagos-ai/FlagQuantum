# API Change Proposal 043: Prospective Twin candidate trial

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after merging evolution-history
alignment.

## Problem

Longitudinal histories show how one Twin behaved over time, but they cannot
answer whether a newly calibrated candidate predicts future QPU measurements
better than the incumbent. Comparing each Twin with a different hardware task
would confound model quality with QPU and finite-shot variation.

## Public API

```python
trial = fq.twin.prepare_candidate_trial(
    incumbent,
    candidate,
    circuit,
    name="candidate-bell",
    shots=1024,
)

# This is the only operation in this workflow that submits a QPU task.
receipt = trial.experiment.submit(provider)
result = provider.fetch_result(receipt)
evaluation = trial.validate_result(
    result,
    receipt=receipt,
    circuit=circuit,
    confidence_level=0.95,
)

print(evaluation.decision)
print(evaluation.candidate_improvement)
print(evaluation.candidate_improvement_lower_bound)
print(evaluation.candidate_improvement_upper_bound)
```

The complete bounded-polling Quafu program is
`examples/remote/quafu_twin_candidate.py`.

## Semantics

Both predictions are frozen before one canonical FlagQuantum circuit is
submitted. The exact same terminal hardware counts are compared with the
incumbent and candidate predictions. A positive `candidate_improvement` means
the candidate has lower total-variation distance to those counts.

Because both comparisons reuse one hardware sample, the evaluation reports a
conservative candidate-improvement uncertainty radius equal to twice the
single-distribution finite-shot TV radius. `improved` or `degraded` is returned
only when the corresponding confidence interval excludes zero; otherwise the
decision is `inconclusive`.

The values compare classical measurement distributions. They are not quantum
state fidelity, amplitude accuracy, causal attribution, or a model-promotion
decision.

## Boundary

Preparing a trial is offline. Only the explicit
`trial.experiment.submit(provider)` call submits one task. The API does not
fetch calibration, poll automatically, retry, submit a second comparison task,
persist artifacts, update a Twin, choose a threshold, promote a model, route a
workload, or expose Agent/MCP behavior.

Validation fails closed unless the incumbent and candidate use the same target,
ordered physical mapping, circuit and ideal baseline; the candidate snapshot is
later; and the receipt, counts and authoritative executed QASM remain bound to
the frozen candidate experiment.

The hardware-validation path currently uses the native Quafu experiment
adapter. Provider-neutral Twins remain constructible and predictable, but a
different provider needs its own reviewed result-binding adapter before it can
use this trial API.

## Compatibility

The function, immutable records and Literal outcomes are additive. Existing
Twin v1 signatures, schemas and metric meanings remain unchanged.
