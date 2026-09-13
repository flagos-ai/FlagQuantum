# API Change Proposal 027: Evidence-bounded Twin assessment

## Status

**Proposed for API-owner review.** The implementation is intentionally additive
and remains in the candidate `flagquantum.twin` namespace. It does not change the
stable root namespace or promote a capability-maturity claim.

## Problem

`QPUDigitalTwin.predict()` produces ideal and calibration-conditioned output
distributions for a frozen circuit. The numerical prediction alone cannot tell a
caller whether later hardware verified that exact circuit, whether an unseen
circuit remains within a measured support envelope, or whether the model should
abstain. Applications currently have to invent those semantics and risk
presenting simulation output as validated accuracy.

## Decisions

- Add the immutable `TwinSupportEnvelope` evidence boundary.
- Add `QPUDigitalTwin.assess(circuit, *, support=None)` and the immutable
  `TwinAssessment` result.
- Add the `TwinDecision` literal with four domain-named outcomes:
  `verified_prediction`, `bounded_estimate`, `physical_reference`, and
  `unsupported`.
- Keep evidence production outside the public model. An envelope records frozen
  identities and externally established bounds; constructing one never creates
  validation evidence.
- Bind every actionable decision to the Twin snapshot, ordered physical-qubit
  mapping, circuit identity or structural limits, and evidence identity.
- Express uncertainty as a total-variation error radius plus its confidence
  level. Do not label the value as state fidelity, amplitude accuracy, or a
  per-shot success probability.
- Keep task submission, polling, model training, promotion policy, and Q-ATLAS
  state machines outside this change.

## Decision semantics

| Decision | Exact later-hardware evidence | Structural support | Error bound | Actionable |
| --- | --- | --- | --- | --- |
| `verified_prediction` | Required | Required | Verified | Yes |
| `bounded_estimate` | Not required | Required | Estimated | Yes |
| `physical_reference` | No actionable evidence | Optional | None | No |
| `unsupported` | No | Failed or mismatched | None | No |

Snapshot or ordered physical-mapping mismatches fail closed as `unsupported`.
A missing envelope returns `physical_reference`, preserving access to the model
output without implying empirical validity.

## Public surface

```python
class QPUDigitalTwin:
    def assess(
        self,
        circuit: Any,
        *,
        support: TwinSupportEnvelope | None = None,
    ) -> TwinAssessment: ...
```

`TwinSupportEnvelope`, `TwinAssessment`, and `TwinDecision` are exported only
from `flagquantum.twin`.

## Compatibility

The proposal is additive. Existing `predict()` and validation behavior remain
unchanged. Serialized support envelopes and assessments use new versioned schema
identifiers. No compatibility alias is introduced.

## Acceptance

- No support envelope yields a non-actionable reference result.
- Exact-circuit evidence with a verified bound yields `verified_prediction`.
- An unseen circuit inside the declared structure with an estimated bound yields
  `bounded_estimate`.
- Snapshot, ordered physical mapping, operation, and size mismatches fail closed.
- Every error radius requires a confidence level.
- Scenario tests cover all four decision paths and deterministic identities.
- Documentation states that the bound applies to measured output distributions,
  not full quantum-state amplitudes.
