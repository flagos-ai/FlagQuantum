# API Change Proposal 061: Regional Twin support-envelope assessment

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice lets an application distinguish an exact released circuit from an unseen
circuit that only fits the release's structural envelope.

## Problem

`TwinRegionRelease.assess(...)` deliberately has two outcomes: an exact frozen
circuit is released, and everything else is outside that exact-circuit release.
That is the right accuracy boundary, but it cannot answer a separate developer
question: *does this unseen circuit at least fit the released region's mapping,
topology, operations, instruction-count and depth envelope?*

Applications should not have to reconstruct that answer from release fields.
More importantly, structural compatibility must not be mistaken for validated
accuracy or used to obtain a prediction carrying borrowed evidence.

## Decision

- Add `TwinRegionRelease.assess_support(region_twin, circuit, *,
  physical_qubits)`.
- Add immutable `TwinRegionSupportAssessment` with exactly three statuses:
  `released_exact_circuit`, `within_envelope_unvalidated`, and
  `outside_envelope`.
- Reuse the release's frozen mapping, topology, operation set, instruction limit,
  depth limit, candidate model identity, snapshot and target. No second support
  artifact or duplicate envelope schema is introduced.
- Return `TwinPrediction` only for `released_exact_circuit`.
  `within_envelope_unvalidated` returns no prediction and the stable reason
  `circuit_identity_not_validated`. `outside_envelope` returns no prediction and
  deterministic identity, mapping, or regional-coverage blockers.
- Keep `TwinRegionRelease.assess(...)`, its two statuses, and every existing
  serialized schema unchanged.

## Public API

```python
import flagquantum as fq

release = fq.twin.load_region_release("region-release.json")
region_twin = fq.twin.load_region_twin("region-twin.json")

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
assessment = release.assess_support(
    region_twin,
    circuit,
    physical_qubits=(20, 27, 34),
)

if assessment.status == "released_exact_circuit":
    prediction = assessment.prediction
elif assessment.status == "within_envelope_unvalidated":
    assert assessment.prediction is None
    print("The circuit fits structurally but still needs future hardware evidence.")
else:
    assert assessment.prediction is None
    print(assessment.reasons)
```

The method is provider-neutral and offline. It never submits, polls, retries,
routes, changes a model, or creates statistical evidence.

## Result contract

| Field | Meaning |
| --- | --- |
| `status` | one of the three evidence-qualified outcomes |
| `release_identity` | identity of the assessed release |
| `circuit_identity` | internally computed circuit content hash |
| `physical_qubits` | assessed ordered logical-to-physical mapping |
| `prediction` | present only for an exact released circuit |
| `reasons` | deterministic refusal or unvalidated reason tokens |
| `schema` | `flagquantum.twin_region_support_assessment.v1` |

The result intentionally has no confidence level, TV-error bound, or routing
authorization. Aggregate release evidence cannot create a per-circuit bound for
an unseen workload.

## Compatibility

Additive. Existing release artifacts, exact-circuit assessments, model
prediction, provider behavior, and contract vocabularies are unchanged. The
Twin v1 contract gains one method, one frozen result type, and one literal type.

## Acceptance

- A verified exact circuit returns `released_exact_circuit` and the same
  prediction as `release.assess(...)`.
- An unseen but structurally covered circuit returns
  `within_envelope_unvalidated`, no prediction, and only
  `circuit_identity_not_validated`.
- Candidate-model, snapshot, target, mapping, topology, operation, instruction,
  depth, or frozen-envelope mismatches return `outside_envelope` with no
  prediction.
- Assessments are deterministic, frozen, offline, provider-neutral, and mutate
  neither model nor release.
- Focused Twin tests, public contract checks, default and runtime PR tiers pass.

## Non-goals

- No arbitrary-circuit accuracy, state fidelity, confidence, uncertainty, or
  TV-error claim.
- No simulation result for structurally compatible but unvalidated circuits.
- No QPU submission, scheduling, retry, routing, model promotion, or trust
  policy.
- No Agent, MCP, natural-language, REST, UI, or provider-specific behavior.
