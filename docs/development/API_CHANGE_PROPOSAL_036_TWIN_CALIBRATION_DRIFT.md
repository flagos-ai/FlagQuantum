# API Change Proposal 036: Twin calibration-drift comparison

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after model persistence.

## Problem

Two frozen Twins can reproduce predictions from two calibration snapshots, but
callers have no provider-neutral way to explain what changed between them.
Applications would otherwise reimplement physical-qubit mapping, time-unit
normalization, readout-distance calculations, and gate-scope matching. That is
error-prone and gives a topology UI no stable source for qubit and coupler drift.

## Public API

```python
import flagquantum as fq

reference = fq.twin.load_twin("shenglian-before.json")
current = fq.twin.load_twin("shenglian-after.json")
drift = fq.twin.compare_calibrations(reference, current)

print(f"max T1 change: {drift.maximum_relative_t1_change:.2%}")
print(f"max T2 change: {drift.maximum_relative_t2_change:.2%}")
if drift.maximum_readout_tv_distance is not None:
    print(f"max readout drift: {drift.maximum_readout_tv_distance:.2%}")
if drift.maximum_relative_gate_duration_change is not None:
    print(
        "max gate-duration change: "
        f"{drift.maximum_relative_gate_duration_change:.2%}"
    )

for qubit in drift.qubit_drifts:
    print(qubit.physical_qubit, qubit.relative_t1_delta)

for gate in drift.gate_duration_drifts:
    print(gate.gate_name, gate.physical_qubits, gate.relative_duration_delta)
```

`compare_calibrations()` is offline and deterministic. It requires the same
provider, backend, ordered physical mapping, logical calibration structure, and
gate-duration scopes. A current snapshot older than the reference fails closed.

## Metric meaning

- T1, T2, and gate-duration relative deltas are `(current - reference) /
  reference`; summary properties report their maximum absolute magnitudes.
- Durations are normalized to seconds before comparison.
- Per-qubit readout drift is the maximum total-variation distance between the
  two conditional readout rows. It is absent when neither Twin has per-qubit
  readout calibration and fails closed if availability changes.
- `channel_model_changed` records a change in noise or readout rules that may
  not be explained by T1, T2, or gate duration alone.
- `has_observed_drift` means at least one represented value changed exactly. It
  is not a statistical-significance decision or an accuracy-loss claim.

The returned `TwinCalibrationDrift`, `TwinQubitCalibrationDrift`, and
`TwinGateDurationDrift` records expose physical qubit and coupler identities so
an application can render topology overlays without reverse-engineering logical
wire order.

## Boundary

This comparison reads two already constructed Twins. It does not fetch
calibration, contact or submit to a QPU, create a history database, set a trust
threshold, update a model, promote evidence, or route a workload. Continuous
refresh and product visualization remain application responsibilities composed
on top of these framework facts.

## Compatibility

The function and three immutable report types are additive. No frozen Twin v1
name, signature, field, status, or existing serialized meaning changes.
