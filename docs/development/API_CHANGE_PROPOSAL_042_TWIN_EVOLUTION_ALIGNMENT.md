# API Change Proposal 042: Twin evolution alignment

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after merging immutable
validation-history extension.

## Problem

Calibration history answers how represented QPU parameters changed. Validation
history answers how measurement-distribution agreement and its evidence bound
changed. Looking at either timeline alone cannot distinguish device movement
from prediction-quality movement, while loosely joining them in an application
risks comparing different snapshots or mappings.

## Public API

```python
import flagquantum as fq

calibration = fq.twin.load_calibration_history("shenglian-calibration-history.json")
validation = fq.twin.load_validation_history("shenglian-validation-history.json")
evolution = fq.twin.align_histories(calibration, validation)

print(evolution.observation_count)
print(evolution.latest_calibration_drift)
print(evolution.latest_twin_agreement_change)
print(evolution.latest_ideal_agreement_change)
print(evolution.latest_qpu_repeatability_change)
print(evolution.latest_verified_bound_change)
```

The checked-in executable example is:

```bash
python examples/twin_evolution_history.py \
  --calibration-history shenglian-calibration-history.json \
  --validation-history shenglian-validation-history.json
```

## Semantics

The alignment requires identical provider, backend, ordered physical mapping,
snapshot identities and capture times. `latest_calibration_drift` is the latest
adjacent-interval drift record. Agreement and bound changes are signed adjacent
differences; unavailable QPU repeatability remains `None` rather than being
invented.

All agreement metrics remain `1 - total variation distance` between classical
measurement distributions. They are not state fidelity or amplitude accuracy.
A synchronized change is not evidence that calibration drift caused the
validation change.

## Boundary

Alignment is deterministic and offline. It does not fetch calibration, submit
or poll QPU work, update or retrain a Twin, estimate causality, choose thresholds,
promote a model, route workloads, persist or publish the aligned view, or expose
Agent/MCP behavior.

## Compatibility

The module-level function and immutable result type are additive. Existing
history schemas, persistence functions, public signatures and metric meanings
remain unchanged.
