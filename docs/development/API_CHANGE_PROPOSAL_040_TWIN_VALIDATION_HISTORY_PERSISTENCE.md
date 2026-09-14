# API Change Proposal 040: Twin validation-history persistence

## Status

**Approved and implemented as a compatible Twin v1 addition.** On 2026-09-14,
the API owner authorized the next Twin increment after merging longitudinal
validation-history construction.

## Problem

`TwinValidationHistory` aligns prediction accuracy evidence across frozen
calibrations, but an in-memory object cannot be reused after a process exits.
Applications need a stable framework artifact without privately reimplementing
schema parsing, integrity checks, or safe-write behavior.

## Public API

The complete offline round trip starts from the already persisted Twins and
validation series:

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
fq.twin.dump_validation_history(history, "shenglian-validation-history.json")

restored = fq.twin.load_validation_history("shenglian-validation-history.json")
print(restored.captured_at)
print(restored.mean_twin_qpu_agreements)
print(restored.mean_ideal_qpu_agreements)
print(restored.mean_qpu_repeatabilities)
print(restored.simultaneous_finite_shot_tv_radii)
print(restored.verified_tv_error_bounds)
```

The executable repository example can create the same artifact:

```bash
python examples/twin_validation_history.py \
  --observation shenglian-state-01.json shenglian-validation-01.json \
  --observation shenglian-state-02.json shenglian-validation-02.json \
  --output shenglian-validation-history.json
```

## Integrity and write semantics

The loader accepts exactly the v1 field set, reconstructs every nested
`TwinValidationSeries`, reruns all longitudinal invariants, and compares the
canonical reconstructed payload with the input. Therefore altered derived
agreement, confidence, shot-radius, or TV-bound arrays fail closed.

The writer creates a credential-free JSON file with mode `0600`. Repeating the
same save is idempotent. A different or invalid existing file is never
overwritten.

## Boundary

Persistence is deterministic and offline. It does not fetch calibration,
submit or poll hardware, retrain a Twin, choose trust thresholds, route
workloads, publish evidence, manage users, or expose Agent/MCP behavior.

## Compatibility

The two module-level functions are additive. The existing history schema,
metrics, serialized Twin and series formats, and all prior public signatures
remain unchanged.
