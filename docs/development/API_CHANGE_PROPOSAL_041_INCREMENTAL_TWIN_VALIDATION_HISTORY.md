# API Change Proposal 041: Incremental Twin validation history

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after merging validation-
history persistence.

## Problem

A quasi-real-time Twin workflow observes later calibration states one at a
time. Rebuilding the entire validation history on every observation is
unnecessary and encourages applications to manipulate serialized internals.
The framework needs one small immutable operation that preserves all existing
evidence bindings.

## Public API

```python
import flagquantum as fq

history = fq.twin.load_validation_history("shenglian-history-state-02.json")
later_twin = fq.twin.load_twin("shenglian-state-03.json")
later_series = fq.twin.load_validation_series("shenglian-validation-03.json")

updated = history.append(later_twin, later_series)
fq.twin.dump_validation_history(updated, "shenglian-history-state-03.json")

print(updated.observation_count)
print(updated.captured_at[-1])
print(updated.mean_twin_qpu_agreements[-1])
print(updated.mean_ideal_qpu_agreements[-1])
print(updated.mean_qpu_repeatabilities[-1])
print(updated.simultaneous_finite_shot_tv_radii[-1])
print(updated.verified_tv_error_bounds[-1])
```

`append()` returns a new `TwinValidationHistory`; the input is never mutated.
The new output uses a new path so the earlier evidence artifact remains intact.

## Invariants

The appended Twin must use the same provider, backend and ordered physical
mapping. Its validation series must be bound to that exact snapshot and retain
the fixed circuit identity and structure. The snapshot must be unique and later
than every existing observation, and hardware-report identities must remain
globally unique across the complete history.

The constructor reruns the same invariants used by initial construction and
loading; there is no weaker incremental path.

## Boundary

Append is deterministic and offline. It does not observe a provider, schedule,
submit or poll QPU work, validate raw provider results, update model parameters,
choose a trust threshold, replace persisted history, route workloads, publish
evidence, or expose Agent/MCP behavior.

## Compatibility

The method is additive on the existing immutable public type. Existing
constructors, functions, schemas, serialized artifacts and metric meanings are
unchanged.
