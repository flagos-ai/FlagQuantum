# API Change Proposal 039: Twin longitudinal validation history

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after merging calibration
history persistence.

## Problem

Calibration drift says how represented device parameters changed; it does not
say how well Twin predictions continued to match hardware. A user needs those
two timelines to remain distinct. Existing `TwinValidationSeries` records the
right metrics for one frozen snapshot, but applications lack a framework-owned
way to align multiple series without risking snapshot, circuit, or report reuse.

## Public API

Each tuple deliberately pairs one saved model with the validation series
produced for its exact snapshot:

```python
import flagquantum as fq

observations = [
    (
        fq.twin.load_twin("shenglian-state-01.json"),
        fq.twin.load_validation_series("shenglian-validation-01.json"),
    ),
    (
        fq.twin.load_twin("shenglian-state-02.json"),
        fq.twin.load_validation_series("shenglian-validation-02.json"),
    ),
]

history = fq.twin.build_validation_history(observations)

for timestamp, twin_match, ideal_match, qpu_repeatability, shot_radius, bound in zip(
    history.captured_at,
    history.mean_twin_qpu_agreements,
    history.mean_ideal_qpu_agreements,
    history.mean_qpu_repeatabilities,
    history.simultaneous_finite_shot_tv_radii,
    history.verified_tv_error_bounds,
    strict=True,
):
    print(timestamp, twin_match, ideal_match, qpu_repeatability, shot_radius, bound)
```

The executable repository example accepts the same persisted inputs:

```bash
python examples/twin_validation_history.py \
  --observation shenglian-state-01.json shenglian-validation-01.json \
  --observation shenglian-state-02.json shenglian-validation-02.json
```

## Invariants

The builder requires at least two observations, strictly increasing and unique
Twin snapshots, the same provider/backend/ordered physical mapping, one fixed
circuit identity and structure, exact series-to-snapshot binding, and globally
distinct hardware-report identities. Repetition count, shots, confidence, and
finite-shot bounds remain visible inside each original validation series.

`TwinValidationHistory.to_dict()` is JSON-ready for an application timeline.
Persistence of this new aggregate is intentionally deferred to a separate
change; the existing Twin and validation-series artifacts remain authoritative
inputs.

## Metric meaning

All agreements are `1 - total variation distance` between classical
measurement distributions. They are not state fidelity, amplitude accuracy,
or the probability that one shot is correct. A verified TV bound includes the
original series' finite-shot uncertainty and is not collapsed into the mean
agreement.

## Boundary

Construction is deterministic and offline. It does not fetch calibration,
submit or poll QPU tasks, update or retrain a model, choose a trust threshold,
route workloads, publish global evidence, or expose agent/MCP behavior.

## Compatibility

The builder and immutable result are additive. Existing Twin APIs, serialized
artifacts, status values, and metric meanings remain unchanged.
