# API Change Proposal 027: QPU Twin evidence reports

## Status

**Proposed for API-owner review.** The implementation is intentionally additive
and remains in the candidate `flagquantum.twin` namespace. It does not change the
stable root namespace or promote a capability-maturity claim.

## Problem

`QPUDigitalTwin.predict()` produces ideal and calibration-conditioned output
distributions for a frozen circuit. The numerical prediction alone cannot tell a
caller whether later hardware verified that exact circuit or whether an unseen
circuit remains within a measured evidence envelope. Applications currently
have to reconstruct those facts and risk presenting simulation output as
validated accuracy.

## Decisions

- Add the immutable `TwinEvidenceEnvelope` evidence boundary.
- Add `QPUDigitalTwin.evidence_report(circuit, *, evidence=None)` and the
  immutable `TwinEvidenceReport` result.
- Add the `TwinEvidenceStatus` literal with four factual outcomes:
  `exact_circuit_verified`, `within_evidence_envelope`, `unverified`, and
  `out_of_scope`.
- Keep evidence production outside the public model. An envelope records frozen
  identities and externally established bounds; constructing one never creates
  validation evidence.
- Bind every bounded report to the Twin snapshot, ordered physical-qubit
  mapping, circuit identity or structural limits, and evidence identity.
- Express uncertainty as a total-variation error bound plus its confidence
  level. Do not label the value as state fidelity, amplitude accuracy, or a
  per-shot success probability.
- Keep task submission, polling, model training, promotion policy, and Q-ATLAS
  state machines outside this change.
- Keep `actionable`, safe-to-use, routing, submission, and approval decisions in
  FlagQAI or another application policy layer. FlagQuantum reports facts only.

## Evidence semantics

| Status | Exact later-hardware evidence | Structural support | Error bound |
| --- | --- | --- | --- |
| `exact_circuit_verified` | Required | Required | Verified |
| `within_evidence_envelope` | Not required | Required | Estimated |
| `unverified` | Not established | Optional | None |
| `out_of_scope` | No | Failed or mismatched | None |

Snapshot or ordered physical-mapping mismatches are reported as `out_of_scope`.
A missing envelope returns `unverified`, preserving access to the model output
without implying empirical validity or deciding what an application should do.

## Public surface

```python
class QPUDigitalTwin:
    def evidence_report(
        self,
        circuit: Any,
        *,
        evidence: TwinEvidenceEnvelope | None = None,
    ) -> TwinEvidenceReport: ...
```

`TwinEvidenceEnvelope`, `TwinEvidenceReport`, and `TwinEvidenceStatus` are
exported only from `flagquantum.twin`.

## Compatibility

The proposal is additive. Existing `predict()` and validation behavior remain
unchanged. Serialized evidence envelopes and reports use new versioned schema
identifiers. No compatibility alias is introduced.

## Acceptance

- No evidence envelope yields an `unverified` report.
- Exact-circuit evidence with a verified bound yields `exact_circuit_verified`.
- An unseen circuit inside the declared structure with an estimated bound yields
  `within_evidence_envelope`.
- Snapshot, ordered physical mapping, operation, and size mismatches report
  `out_of_scope`.
- Every error bound requires a confidence level.
- The public result contains no actionable, routing, or approval field.
- Scenario tests cover all four evidence paths and deterministic identities.
- Documentation states that the bound applies to measured output distributions,
  not full quantum-state amplitudes.
