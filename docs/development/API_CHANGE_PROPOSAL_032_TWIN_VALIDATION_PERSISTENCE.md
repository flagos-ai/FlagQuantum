# API Change Proposal 032: Twin validation-series persistence

## Status

**Implementation authorized; pending API freeze review.** After Proposal 031
merged, the user authorized the next Twin slice: preserve the repeated
validation metrics required by downstream products without making them depend
on a live Python process.

## Problem

`TwinValidationSeries.to_evidence()` intentionally reduces repeated
observations to an exact-circuit support envelope and conservative error bound.
That envelope does not retain the separate Twin-to-QPU, ideal-to-QPU, QPU
repeatability, repetition, and shot-count metrics. Downstream products need a
canonical offline artifact to display and audit those values.

## Decision

- Add `TwinValidationSeries.from_dict(...)` with an exact version-1 field set
  and the existing constructor invariants.
- Add `dump_validation_series(...)` and `load_validation_series(...)` in
  `flagquantum.twin`.
- Write canonical, sorted JSON by exclusive file creation.
- Treat saving the same series to the same path as an idempotent no-op.
- Refuse to replace a different or invalid artifact.
- Perform no provider access, submission, polling, calibration retrieval,
  model update, evidence promotion, routing, or application publication.

## Public API

```python
series = experiment.validation_series(
    [first_hardware_report, second_hardware_report],
    circuit=circuit,
    confidence_level=0.95,
)

fq.twin.dump_validation_series(series, "twin-validation.json")
restored = fq.twin.load_validation_series("twin-validation.json")

print(f"Twin ↔ QPU: {restored.mean_twin_qpu_agreement:.2%}")
print(f"Ideal SV ↔ QPU: {restored.mean_ideal_qpu_agreement:.2%}")
if restored.mean_qpu_repeatability is not None:
    print(f"QPU repeatability: {restored.mean_qpu_repeatability:.2%}")
```

The complete executable Quafu workflow remains
`examples/remote/quafu_twin_evidence.py`. It now saves and reloads both the
validation series and its reduced evidence envelope.

## Boundary and interpretation

The artifact is a canonical summary, not a signed provider receipt and not an
independently reproducible bundle of raw counts. Its identity changes when any
serialized value changes. The agreement fields remain `1 - total variation`
for classical measurement distributions; they are not quantum-state fidelity,
amplitude accuracy, or per-shot success probability.

Persistence does not make evidence global. FlagQuantum defines the portable
artifact; storage, publication, tenancy, access control, dashboards, and drift
curves remain responsibilities of applications such as FlagQAI.

## Compatibility

The change is additive. Existing prediction, experiment, hardware-report,
evidence, and validation-series schemas retain their meanings. The Stable Core
root manifest does not change.

## Acceptance

- A validation series round-trips without changing its value or identity.
- Re-saving the same series is idempotent.
- A different or invalid existing file is never replaced.
- Missing, malformed, non-object, extra-field, and incomplete payloads fail
  closed.
- Loading is deterministic and performs no provider call.
- The public contract, documentation, and complete executable example cover
  both new functions.
