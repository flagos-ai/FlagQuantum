# API Change Proposal 027: QPU Twin evidence reports

## Status

**Implementation authorized; pending API freeze review.** On 2026-09-13, an
explicit user directive established `fq.twin` as the framework entry point for
constructing digital twins of arbitrary QPUs, with native Quafu support. The
change remains additive and does not promote a capability-maturity claim.

## Framework boundary

- `fq.twin` is a provider-neutral FlagQuantum framework namespace.
- Any QPU integration can build a Twin from a FlagQuantum device-backed
  `NoiseModel` through `fq.twin.from_noise_model(...)`.
- Quafu calibration conversion and provider access are supported natively, but
  Quafu does not define the general Twin abstraction.
- This repository contains Python modeling, prediction, evidence, validation,
  and provider integration APIs. It contains no natural-language interface,
  agent protocol, MCP server, or application policy.

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
- Keep actionable, safe-to-use, routing, and approval decisions outside the
  Twin model and evidence-report API. FlagQuantum reports facts only.

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
def from_noise_model(
    noise_model: NoiseModel,
    *,
    provider: str,
    backend: str,
    qubits: Sequence[int],
) -> QPUDigitalTwin: ...

def from_quafu_chip_info(
    chip_info: Mapping[str, Any],
    *,
    backend: str,
    qubits: Sequence[int],
    readout_confusion_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
    correlated_readout_confusion_matrix: Sequence[Sequence[float]] | None = None,
) -> QPUDigitalTwin: ...

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

The module itself is available as the lazily loaded `fq.twin` root namespace.
The short construction functions live inside that namespace; no Twin class or
provider-specific constructor is flattened into the package root.
This narrowly supersedes Proposal 017's decision not to add a root namespace;
all other Proposal 017 ownership and maturity boundaries remain unchanged.

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
- `import flagquantum as fq` exposes the generic API through `fq.twin`.
- Generic construction and native Quafu construction are tested separately.
- The short construction functions are equivalent to the corresponding class
  factories and have machine-checked signatures.
- Scenario tests cover all four evidence paths and deterministic identities.
- Documentation states that the bound applies to measured output distributions,
  not full quantum-state amplitudes.
