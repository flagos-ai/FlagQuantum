# API Change Proposal 055: Regional Twin holdout evolution

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice aligns existing calibration and regional holdout histories without
collecting or changing evidence.

## Problem

`TwinCalibrationHistory` reports how comparable QPU calibrations changed, while
`TwinRegionHoldoutHistory` reports how the same predeclared reference and
holdout circuits agreed with later hardware. Consumers currently have to align
their snapshot identities and derive changes themselves before drawing a drift
curve. That duplicates an evidence-sensitive invariant outside the framework.

## Decision

- Add `TwinRegionHoldoutEvolution` and
  `align_region_holdout_history(...)`.
- Require exact provider/backend, physical mapping, snapshot-identity sequence,
  and capture-time sequence agreement between both source histories.
- Expose adjacent changes in reference agreement, holdout agreement, holdout
  TV-error increase, and the holdout simultaneous error bound.
- Expose interval maximum relative T1, T2, gate-duration, and readout-TV drift
  directly from the authoritative calibration history.
- Preserve both complete source histories in the immutable result.
- Perform no provider access, task submission, threshold selection, causal
  inference, model update, promotion, or workload routing.

## Public API

```python
import flagquantum as fq

calibration_history = fq.twin.load_calibration_history(
    "region-calibration-history.json"
)
holdout_history = fq.twin.load_region_holdout_history(
    "region-holdout-history.json"
)

evolution = fq.twin.align_region_holdout_history(
    calibration_history,
    holdout_history,
)

print(evolution.interval_maximum_relative_t1_changes)
print(evolution.interval_maximum_readout_tv_distances)
print(evolution.holdout_twin_qpu_agreement_changes)
print(evolution.holdout_simultaneous_tv_error_bound_changes)
```

## Compatibility

The function, type, and schema are additive. Existing APIs and serialized
artifacts remain unchanged. The evolution result is JSON-ready through
`to_dict()` but does not introduce a second persistence format for its already
persisted source histories.

## Acceptance

- Exact matching histories align deterministically.
- Target, mapping, snapshot, or capture-time mismatch fails closed.
- Derived arrays cover every adjacent interval and retain unavailable drift as
  `None` rather than inventing values.
- The payload is finite JSON and contains no causal or promotion decision.
- Documentation includes a complete offline example.
