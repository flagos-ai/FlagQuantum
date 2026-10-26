# API Change Proposal 096: Twin evidence freshness assessment

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice lets an application apply explicit age, drift, and verified-error
thresholds to an existing frozen Twin and its offline histories.

## Problem

FlagQuantum can preserve calibration and validation histories, but it
deliberately chooses no trust threshold. Applications therefore have to
reconstruct target, mapping, chronology, drift, age, and error-bound checks.
That duplicates evidence logic and makes it easy to present a stale validation
bound as if it applied to the current snapshot.

The framework needs a pure policy assessment. It must not invent universal
thresholds, infer causal decay, refresh a model, contact a provider, or
authorize routing.

## Decision

- Add immutable `TwinFreshnessPolicy` containing only caller-supplied
  thresholds.
- Add immutable `TwinFreshnessAssessment` containing the evaluated snapshot,
  relevant existing drift and validation records, derived ages, next age-based
  deadline, and deterministic reasons.
- Add `TwinFreshnessStatus` with exactly four values:
  `current_under_policy`, `refresh_due`, `insufficient_evidence`, and
  `incompatible_history`.
- Add `TwinFreshnessPolicy.assess(twin, *, calibration_history=None,
  validation_history=None, as_of)` as an offline operation.
- Reuse `TwinCalibrationDrift`, `TwinCalibrationHistory`,
  `TwinValidationSeries`, and `TwinValidationHistory`. Introduce no second drift,
  validation, or evidence representation.
- Do not serialize the policy or assessment in this slice. Applications may
  persist the authoritative source histories and evaluate a policy again.

## Time semantics

`TwinValidationSeries` and `TwinValidationHistory` do not carry an authoritative
provider result-completion time. Their `captured_at` value is the calibration
capture time of the Twin that was validated.

Consequently, `maximum_validation_age` and
`validation_snapshot_age_seconds` refer to the age of the calibration snapshot
associated with the latest validation series. They do not claim to represent
task submission, execution, return, ingestion, or report time. Adding a
provider-attested result timestamp would require a separate versioned evidence
contract.

## Public API

```python
from datetime import datetime, timedelta, timezone

import flagquantum as fq

policy = fq.twin.TwinFreshnessPolicy(
    maximum_calibration_age=timedelta(hours=4),
    maximum_validation_age=timedelta(hours=8),
    maximum_relative_t1_drift=0.20,
    maximum_relative_t2_drift=0.20,
    maximum_readout_tv_drift=0.02,
    maximum_relative_gate_duration_drift=0.10,
    maximum_verified_tv_error_bound=0.10,
)
assessment = policy.assess(
    current_twin,
    calibration_history=calibration_history,
    validation_history=validation_history,
    as_of=datetime.now(timezone.utc),
)

print(assessment.status)
print(assessment.reasons)
print(assessment.next_refresh_at)
```

All thresholds are optional, but at least one is required. Age thresholds must
be positive `datetime.timedelta` values. Drift thresholds are finite and
non-negative; readout TV and verified TV-error thresholds are also at most one.
`as_of` is mandatory and timezone-aware so repeated assessments are
deterministic.

## Result contract

| Field | Meaning |
| --- | --- |
| `status` | one of the four policy-qualified outcomes |
| `snapshot_identity` | identity of the assessed current Twin |
| `evaluated_at` | caller-supplied, timezone-aware assessment time |
| `calibration_age_seconds` | age of the current Twin calibration snapshot |
| `validation_snapshot_age_seconds` | age of the latest validated calibration snapshot, when supplied |
| `next_refresh_at` | earliest age-based deadline, or evaluation time when already due |
| `calibration_drift` | existing latest interval drift record, when applicable |
| `validation_series` | existing latest validation series, when applicable |
| `reasons` | deterministic due, missing-evidence, or incompatibility facts |

The assessment deliberately has no prediction, confidence reinterpretation,
routing authorization, promotion action, provider handle, or mutable state.

## Status semantics

- `current_under_policy`: every configured threshold with available applicable
  evidence passes.
- `refresh_due`: at least one configured threshold is exceeded, or the latest
  validation series belongs to an older snapshot.
- `insufficient_evidence`: a configured threshold requires a missing history or
  unavailable drift component.
- `incompatible_history`: target, ordered mapping, current snapshot, or
  caller-supplied time is incompatible with the assessment.

Incompatibility takes precedence over insufficient evidence, which takes
precedence over a due threshold. This prevents a partial policy pass from
concealing a stronger blocker.

## Compatibility

Additive. No existing Twin class, method, status, serialized schema, provider
behavior, or result meaning changes. The new records are runtime-only and do
not alter any v1 artifact.

## Acceptance

- A current matching Twin with all configured thresholds satisfied returns
  `current_under_policy` and no reasons.
- Exceeded age, drift, or verified-bound thresholds return `refresh_due` with
  deterministic reason tokens.
- Validation for an older snapshot returns `refresh_due` and
  `current_snapshot_not_validated`.
- Required missing histories or drift fields return `insufficient_evidence`.
- Target, mapping, current-snapshot, or time mismatches return
  `incompatible_history`.
- The assessment is deterministic for the same inputs and explicit `as_of`.
- The operation performs no provider I/O and mutates no input.
- Focused Twin tests, public API contracts, documentation checks, strict typing,
  and default CI tiers pass.

## Non-goals

- No universal freshness or trust thresholds.
- No provider result-completion timestamp.
- No causal calibration-to-accuracy model or inferred validity duration.
- No calibration collection, task scheduling, submission, polling, retry, or
  cancellation.
- No model fitting, mutation, promotion, rollback, routing, or hardware control.
- No Agent, MCP, natural-language, REST, UI, dashboard, or database behavior.
