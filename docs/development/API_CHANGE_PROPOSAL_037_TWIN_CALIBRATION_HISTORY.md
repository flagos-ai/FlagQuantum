# API Change Proposal 037: Twin calibration history

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner requested the next Twin increment after calibration comparison.

## Problem

`compare_calibrations()` answers what changed between two frozen Twins. A drift
curve needs the same calculation across a chronological sequence, with a clear
distinction between long-term movement from the original baseline and the most
recent step. Reimplementing chronology, identity, and comparability checks in
each application would make those curves inconsistent.

## Public API

The complete offline workflow uses model artifacts produced by
`fq.twin.dump_twin()`:

```python
import flagquantum as fq

twins = [
    fq.twin.load_twin("shenglian-state-01.json"),
    fq.twin.load_twin("shenglian-state-02.json"),
    fq.twin.load_twin("shenglian-state-03.json"),
]

history = fq.twin.build_calibration_history(twins)

print(history.provider, history.backend_name, history.physical_qubits)
for timestamp, cumulative, incremental in zip(
    history.captured_at[1:],
    history.baseline_drifts,
    history.interval_drifts,
):
    print(
        timestamp,
        f"T1 from first: {cumulative.maximum_relative_t1_change:.2%}",
        f"T1 since previous: {incremental.maximum_relative_t1_change:.2%}",
    )
```

`baseline_drifts[i]` compares the first snapshot with snapshot `i + 1`.
`interval_drifts[i]` compares snapshot `i` with snapshot `i + 1`. Each nested
record is the existing `TwinCalibrationDrift`, so its per-qubit and per-gate
physical scopes are directly reusable by topology and chart applications.

## Validation

The builder requires at least two `QPUDigitalTwin` objects, unique snapshot
identities, and strictly increasing `captured_at` values. Every comparison must
identify the same provider, backend, ordered physical mapping, logical
calibration structure, readout availability, and gate-duration scopes.

`TwinCalibrationHistory.to_dict()` returns a JSON-ready value. Persistence of a
long-lived history store is intentionally not introduced in this change.
The executable `examples/twin_calibration_history.py` accepts two or more saved
Twin files and prints the same cumulative and interval series.

## Boundary

History construction is deterministic and offline. It does not fetch or
schedule calibration snapshots, contact a provider, submit hardware work,
persist a database, define a trust threshold, infer prediction accuracy,
retrain a Twin, route a circuit, or expose an agent protocol.

## Compatibility

The builder and immutable history record are additive. Existing Twin v1 names,
signatures, artifacts, statuses, and meanings remain unchanged.
