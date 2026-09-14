# API Change Proposal 038: Twin calibration-history persistence

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after merging calibration
history construction.

## Problem

`build_calibration_history()` produces the framework facts needed by a drift
curve, but its result currently disappears with the Python process. Requiring
each application to invent serialization would weaken schema, chronology,
identity, and derived-summary validation.

## Public API

This complete offline example begins with Twin model artifacts previously
created by `fq.twin.dump_twin()`:

```python
import flagquantum as fq

twins = [
    fq.twin.load_twin("shenglian-state-01.json"),
    fq.twin.load_twin("shenglian-state-02.json"),
    fq.twin.load_twin("shenglian-state-03.json"),
]
history = fq.twin.build_calibration_history(twins)

fq.twin.dump_calibration_history(history, "shenglian-history.json")
restored = fq.twin.load_calibration_history("shenglian-history.json")

for timestamp, cumulative, incremental in zip(
    restored.captured_at[1:],
    restored.baseline_drifts,
    restored.interval_drifts,
):
    print(
        timestamp,
        f"T1 from first: {cumulative.maximum_relative_t1_change:.2%}",
        f"T1 since previous: {incremental.maximum_relative_t1_change:.2%}",
    )
```

The repository CLI exposes the same workflow:

```bash
python examples/twin_calibration_history.py \
  shenglian-state-01.json shenglian-state-02.json shenglian-state-03.json \
  --output shenglian-history.json
```

## File semantics

The canonical JSON stores the history schema, QPU target, ordered physical
mapping, snapshot identities and timestamps, plus the cumulative and interval
`TwinCalibrationDrift` records. Loading reconstructs every immutable record and
rejects unknown, missing, non-canonical, inconsistent, or tampered fields,
including derived summaries.

The writer creates a private mode-0600 file. Repeating an identical write is
idempotent. Existing different or invalid content is not replaced.

## Boundary

The artifact contains no credentials, provider client, source Twin models, QPU
task, prediction evidence, trust threshold, or policy decision. Loading and
writing perform no network operation. Long-lived storage, scheduling,
publication, visualization, model updates, and agent/MCP behavior remain
application responsibilities.

## Compatibility

The two persistence functions are additive. The existing history schema and
all earlier Twin APIs and artifacts retain their meanings.
