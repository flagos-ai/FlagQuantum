# API Change Proposal 049: Prospective Twin-region validation

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice follows connected regional-model composition with an explicit path from a
regional prediction to repeated, identity-bound QPU evidence.

## Problem

`TwinRegionModel` can predict a wider circuit but intentionally carries no
regional accuracy claim. Calling `TwinExperiment.prepare(region_twin.twin, ...)`
directly would bypass the region's mapping and topology gate. Applications also
need one fail-closed way to turn repeated results into exact-circuit regional
support without treating unexercised couplers as validated.

## Decision

- Add `TwinRegionModel.prepare_experiment(...)`.
- Require the complete physical mapping to equal the model's canonical regional
  wire order and require the circuit to pass region coverage before freezing.
- Reuse `TwinExperiment` unchanged for canonical OpenQASM, explicit provider
  submission, receipt validation, and repeated validation series.
- Add `TwinRegionModel.support_from_validation_series(...)`.
- Require at least two distinct bound hardware tasks, the same composed Twin
  snapshot, exact circuit identity, and exact physical mapping.
- Return an existing `TwinCircuitSupport` whose evidence is exact-circuit
  evidence and whose directed couplers are only those exercised by the circuit.
- Do not validate every circuit or every coupler in the composed region, infer
  correlated noise, submit automatically, retry, promote a model, or add an
  agent/MCP interface.

## Public API

```python
experiment = region_twin.prepare_experiment(
    circuit,
    physical_qubits=(20, 27, 34),
    name="regional-ghz",
    shots=1024,
)

first_receipt = experiment.submit(provider)
first_result = provider.fetch_result(first_receipt)
first_report = experiment.validate_result(first_result, receipt=first_receipt)

second_receipt = experiment.submit(provider)
second_result = provider.fetch_result(second_receipt)
second_report = experiment.validate_result(second_result, receipt=second_receipt)

series = experiment.validation_series(
    (first_report, second_report),
    circuit=circuit,
)
support = region_twin.support_from_validation_series(
    series,
    circuit,
    physical_qubits=(20, 27, 34),
)
report = support.evidence_report(region_twin.twin, circuit)

print(report.status)
print(series.mean_twin_qpu_agreement)
print(series.mean_ideal_qpu_agreement)
print(series.mean_qpu_repeatability)
print(report.tv_error_bound)
print(report.confidence_level)
```

The complete polling and persistence workflow is
`examples/remote/quafu_twin_region_validation.py`. It submits exactly two tasks
only when the user runs it with a configured Quafu token.

## Compatibility

The two methods are additive. Existing serialized schemas, standalone
`TwinExperiment`, validation-series behavior, evidence meanings, and public
module names remain unchanged.

## Acceptance

- Experiment preparation fails before provider I/O for an incorrect mapping or
  circuit outside the regional topology and limits.
- Two distinct bound results can produce exact-circuit regional support.
- One result, a foreign snapshot, changed mapping, changed circuit, rewritten
  program, or duplicate task fails closed.
- Only couplers exercised by the exact validated circuit appear in the returned
  support artifact.
- Documentation and the executable example call the percentages agreement of
  classical measurement distributions, not state fidelity.
