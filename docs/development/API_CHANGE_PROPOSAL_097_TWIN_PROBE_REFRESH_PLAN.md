# API Change Proposal 097: Deterministic Twin probe-refresh plans

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice converts a caller-supplied circuit pool into a bounded, deterministic,
offline plan for collecting later regional validation evidence.

## Problem

FlagQuantum can assess release coverage, calibration drift, validation history,
and evidence freshness, but an application still has to choose which fixed
circuits to validate next. Ad hoc selection can exceed a hardware budget,
silently replace unsupported circuits, or lose the identities needed to explain
why a later result was collected.

The framework needs a pure planning operation. It must select only from the
circuits supplied by the caller, bind the existing release and histories, and
reuse the existing regional validation-suite path. It must not schedule,
submit, poll, retry, cancel, or infer that refreshed evidence will be accurate.

## Decision

- Add immutable `TwinProbeRefreshPlan`.
- Add `prepare_probe_refresh(release, candidate_circuits, *,
  calibration_history=None, validation_history=None, maximum_tasks, shots)`.
- Use exactly two planned repetitions per selected circuit, matching the
  minimum repeated-task requirement of `TwinRegionValidationSuite`.
- Rank structurally covered candidates by this deterministic priority:
  absent from the exact release, absent from supplied validation history,
  overlap with the latest observed calibration drift, directed-coupler count,
  instruction count, then circuit identity.
- Report structurally unsupported candidates with their existing regional
  coverage reason tokens. Never replace one with a generated circuit.
- Add `plan.prepare_suite(region_twin, candidate_circuits, *, name)` as the
  only materialization path. It selects the identity-bound circuits and calls
  the existing `prepare_region_validation_suite(...)` implementation.
- Keep the plan runtime-only in this slice. `to_dict()` and `identity` provide a
  canonical representation, but no file persistence API is introduced.

## Public API

```python
import flagquantum as fq

release = fq.twin.load_region_release("region-release.json")
region_twin = fq.twin.load_region_twin("region-twin.json")
calibration_history = fq.twin.load_calibration_history(
    "calibration-history.json"
)
validation_history = fq.twin.load_region_validation_history(
    "region-validation-history.json"
)

candidate_circuits = (
    fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
    fq.Circuit(3).x(2).cx(1, 2),
    fq.Circuit(3).h(2).cx(0, 1),
)

plan = fq.twin.prepare_probe_refresh(
    release,
    candidate_circuits,
    calibration_history=calibration_history,
    validation_history=validation_history,
    maximum_tasks=6,
    shots=1024,
)

print(plan.selected_circuit_identities)
print(plan.covered_directed_couplers)
print(plan.coverage_gaps)
print(plan.planned_task_count)

# This remains offline. It freezes existing TwinExperiment objects but does not
# submit them. Each later provider submission is still explicit.
suite = plan.prepare_suite(
    region_twin,
    candidate_circuits,
    name="regional-refresh",
)
assert suite.planned_task_count == plan.planned_task_count
```

Candidate circuits map logical qubits to `release.physical_qubits` in order.
The release therefore supplies the target, mapping, topology, supported
operations, instruction limit, and depth limit. The application supplies the
candidate pool and hardware budget.

## Ranking semantics

Ranking identifies an evidence gap; it does not predict scientific value,
accuracy gain, or causal model improvement. A higher-ranked circuit is only a
better fit under the explicit ordering above.

Calibration overlap uses represented changes in the latest interval drift. It
does not claim statistical significance or that the change caused a prediction
error. Validation history records whether a circuit identity was previously
observed; it does not transfer a TV bound from one snapshot to another.

Circuit identity is the final tie-breaker, so candidate input order cannot alter
the plan. The two-repetition rule means a task budget of `N` can select at most
`N // 2` circuits. A budget too small for a regional suite returns an explained
zero- or one-circuit plan; `prepare_suite(...)` then fails closed because the
existing suite contract requires at least two circuits.

## Result contract

The plan binds:

- release, candidate-region, and candidate-snapshot identities;
- canonical target and ordered physical mapping;
- every distinct candidate circuit identity and the selected subset;
- unsupported circuit identities and regional coverage reasons;
- selected directed couplers and deterministic coverage-gap tokens;
- canonical calibration- and validation-history identities, when supplied;
- shots, two repetitions, task budget, planned task count, and total shots.

The plan contains no provider client, credential, task handle, queue state,
retry policy, routing authorization, promotion action, or mutable model.

## Compatibility

Additive. No existing Twin class, method, status, provider behavior, release
meaning, or serialized v1 artifact changes. The Twin v1 contract gains one
factory, one frozen runtime result, two methods, and three derived properties.

## Acceptance

- Identical release, histories, candidate identities, budget, and shots produce
  byte-identical canonical plan payloads and the same identity.
- Reordering candidate inputs does not change the plan.
- Unsupported candidates are reported with existing coverage reasons and are
  never selected or substituted.
- Duplicate circuit identities fail closed.
- Calibration and validation histories must match the release target and
  ordered mapping; calibration history must contain the release snapshot.
- Planned tasks never exceed `maximum_tasks`.
- Suite materialization rejects model, snapshot, target, mapping, or selected
  candidate mismatches and delegates to the existing regional suite builder.
- Planning and materialization perform no provider or network operation.
- Focused Twin tests, public API contracts, strict typing, documentation checks,
  default CI, and runtime CI pass.

## Non-goals

- No generated probes, circuit synthesis, global optimization, active learning,
  parameter identifiability, fitting, or automatic model evolution.
- No claim that the selected probes are sufficient to identify a model.
- No universal task budget, shots, freshness policy, or drift threshold.
- No QPU submission, polling, retry, cancellation, scheduling, or routing.
- No Agent, MCP, natural-language, REST, UI, dashboard, or provider-specific
  behavior.
