# API Change Proposal 059: Release-bound circuit assessment

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice turns a `TwinRegionRelease` into a per-circuit verdict for one explicit
circuit, and adds no authorization beyond the release recorded in PR #167.

## Problem

A released regional Twin can name the exact circuits it was qualified on, but an
application still lacks one authoritative call answering the operational
question: *is this exact circuit on this regional model covered by this
release?* Re-deriving that answer from release fields in application code would
duplicate the release's identity rules and would let a caller pair a prediction
with a release, snapshot, target, ordered mapping, or circuit identity that the
release never covered. Treating aggregate candidate-improvement evidence as a
per-circuit accuracy statement would also exceed the recorded evidence.

## Decision

- Add `TwinRegionRelease.assess(region_twin, circuit, *, physical_qubits)` and
  the immutable result type `TwinReleaseAssessment` with the status vocabulary
  `released_exact_circuit | outside_release`.
- Require simultaneous agreement of release identity, candidate regional model
  identity, candidate snapshot identity, provider/backend target, ordered
  physical mapping, structural coverage, and exact frozen circuit identity.
  Every identity is computed internally; callers never supply fingerprints.
- Report all mismatches together as deterministic reason tokens, including the
  regional coverage tokens, so one refusal explains the complete decision.
- Return a `TwinPrediction` only for `released_exact_circuit`, produced by the
  authoritative `TwinRegionModel.predict` path, so the prediction carries the
  released snapshot and circuit identity.
- Add no serialized artifact, no confidence level, no total-variation error
  bound, no routing authorization, no provider semantics, and no model or
  release mutation.

## Justification for a new result type

The existing authoritative result type for "does the Twin support this circuit"
is `TwinEvidenceReport`. It cannot express this requirement without becoming
misleading:

| Required fact | `TwinEvidenceReport` today |
| --- | --- |
| release identity | no field; `evidence_envelope_identity` identifies a `TwinEvidenceEnvelope`, not a `TwinRegionRelease` |
| candidate regional model identity | no field |
| provider/backend target | no field |
| ordered physical mapping verdict | no field (`TwinRegionCoverage` is a separate type) |
| statistical fields | requires `tv_error_bound` and `confidence_level` to appear together for `exact_circuit_verified` |

Reusing `TwinEvidenceReport` would therefore either overload an unrelated
identity field or force a release assessment to fabricate an error bound that
the release evidence does not support. `TwinRegionCoverage` is reused unchanged
for structural facts, and `TwinPrediction` is reused unchanged for the released
prediction. `TwinReleaseAssessment` is the smallest type that states the missing
release-binding facts explicitly.

## Public API

```python
release = fq.twin.load_region_release("region-release.json")
assessment = release.assess(
    region_twin,
    circuit,
    physical_qubits=(20, 27, 34),
)

print(assessment.status)
print(assessment.prediction)
print(assessment.reasons)
print(assessment.release_identity)
```

`region_twin` is a composed regional model, for example from
`fq.twin.compose_region_twin(...)` or an application's own loader. The call
performs no provider I/O, submits nothing, mutates nothing, and never routes.

## Result contract

| Field | Meaning |
| --- | --- |
| `status` | `released_exact_circuit` or `outside_release` |
| `release_identity` | identity of the assessed `TwinRegionRelease` |
| `circuit_identity` | frozen content hash of the assessed circuit |
| `physical_qubits` | ordered logical-to-physical mapping that was assessed |
| `prediction` | released `TwinPrediction`, or `None` |
| `reasons` | unique deterministic reason tokens, empty on success |
| `schema` | `flagquantum.twin_release_assessment.v1` |

Stable reason tokens: `candidate_region_identity_mismatch`,
`snapshot_identity_mismatch`, `target_mismatch`, `physical_mapping_mismatch`,
`circuit_identity_outside_release`, plus the regional coverage tokens
`physical_qubits_outside_region`, `physical_couplers_outside_region`,
`operations_outside_region_support`, `multi_qubit_operation_outside_support`,
`maximum_instruction_count_exceeded`, and `maximum_circuit_depth_exceeded`.

## Compatibility

Additive. The released `TwinRegionRelease` payload, the Twin v1 schemas, and all
existing model, candidate, holdout, evidence, submission, prediction, and
provider behavior are unchanged. No new serialized artifact is introduced.

## Acceptance

- A structurally covered circuit whose frozen identity is in the release
  returns `released_exact_circuit` with a prediction bound to the release.
- A circuit that is structurally covered but not in the release returns
  `outside_release`, `prediction is None`, and
  `("circuit_identity_outside_release",)`.
- A wrong candidate regional model, wrong snapshot, tampered release snapshot
  identity, wrong target, wrong ordered mapping, and structural mismatch each
  fail closed with a deterministic reason and no prediction.
- Consecutive assessments are equal, frozen, and mutate neither the release nor
  the model; the release still reports `routing_authorized is False`.

## Non-goals

- No arbitrary-circuit accuracy claim, no per-circuit confidence level, and no
  total-variation error bound derived from aggregate improvement evidence.
- No workload routing, plan rewriting, task submission, polling, or retry.
- No mutation of the release, the regional model, or an application's active
  model.
- No Agent, MCP, natural-language, or application-policy surface.
- No provider-specific behavior; no Quafu or other vendor semantic in core Twin
  behavior.
