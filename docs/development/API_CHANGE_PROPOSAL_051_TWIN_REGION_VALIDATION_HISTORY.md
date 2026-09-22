# API Change Proposal 051: Regional Twin validation history

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice records the results of one fixed regional validation suite across
strictly ordered calibration snapshots.

## Problem

A regional suite evaluation describes one calibration snapshot. Applications
need a canonical, auditable way to compare the same fixed circuit suite on the
same QPU region over time without treating unrelated circuits, mappings, or
topologies as one longitudinal series.

## Decision

- Add immutable `TwinRegionValidationHistory` and
  `build_region_validation_history(...)`.
- Require at least two evaluations with the same provider, backend, physical
  mapping, full directed topology, ordered circuit identities, exercised
  couplers, and maximum circuit depth.
- Require strictly increasing calibration timestamps and globally distinct
  hardware-report identities.
- Record calibration and region identities for every observation so that
  calibration changes remain explicit evidence rather than hidden model
  updates.
- Expose chart-ready agreement, repeatability, uncertainty, task, and shot
  sequences without defining a trust threshold.
- Support immutable `append(...)` and strict canonical persistence through
  `dump_region_validation_history(...)` and
  `load_region_validation_history(...)`.
- Keep construction and persistence entirely offline. The history never
  submits, polls, retries, routes, promotes, or mutates a Twin.

## Public API

```python
import flagquantum as fq

# Each evaluation was already produced by its corresponding frozen regional
# validation suite after explicit, application-controlled hardware execution.
history = fq.twin.build_region_validation_history(
    [
        (reference_region_twin, reference_evaluation),
        (current_region_twin, current_evaluation),
    ]
)

print(history.mean_twin_qpu_agreements)
print(history.mean_ideal_qpu_agreements)
print(history.mean_qpu_repeatabilities)
print(history.simultaneous_tv_error_bounds)
print(history.total_shots)

fq.twin.dump_region_validation_history(
    history,
    "twin-region-validation-history.json",
)
restored = fq.twin.load_region_validation_history(
    "twin-region-validation-history.json"
)

updated = restored.append(later_region_twin, later_evaluation)
```

## Compatibility

The new functions and type are additive. Existing Twin, regional model,
regional suite, evidence, and validation-history signatures and serialized
meanings do not change.

## Acceptance

- Construction rejects changed targets, mappings, full topologies, suite
  circuits, suite structure, non-increasing timestamps, and reused reports.
- Loading validates every nested evaluation and recomputes derived arrays.
- Persistence is private, canonical, create-once, and idempotent only for
  identical content.
- Metrics describe classical measurement distributions, not quantum-state
  fidelity or arbitrary-circuit accuracy.
- No provider I/O, model evolution, trust policy, or promotion is introduced.
