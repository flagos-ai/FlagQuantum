# API Change Proposal 031: Repeated Twin validation series

## Status

**Implementation authorized; pending API freeze review.** After Proposal 030
merged, the user authorized the next Twin slice: separate prediction error,
ideal-baseline error, hardware repeatability, and finite-shot uncertainty across
distinct executions of one frozen experiment.

## Problem

One `TwinHardwareReport` can prove identity binding for one execution, but it
cannot distinguish a stable prediction error from one noisy hardware sample.
`TwinEvidenceEnvelope` records a support boundary and final error radius; it is
not the authoritative representation of repeated observations. Changing either
existing schema to contain a variable series would mix responsibilities and
break their frozen version-1 meanings.

## Decision

- Add immutable `TwinValidationSeries` as the summary of distinct,
  identity-bound reports for one `TwinExperiment` and circuit.
- Add `TwinExperiment.validation_series(...)` as the shortest construction
  path. It reuses every fail-closed circuit, program, result, and physical
  mapping check from `evidence_from_report()`.
- Report mean and maximum Twin-to-QPU TV distance separately from mean and
  maximum ideal-to-QPU TV distance.
- Report mean and maximum pairwise QPU observation distance when two or more
  repetitions exist. This observed repeatability includes finite-shot noise and
  is not an intrinsic hardware-fidelity claim.
- Use a Bonferroni-adjusted per-report finite-shot radius so the final maximum
  error bound has the requested simultaneous confidence across all reports.
- Reject an empty series, duplicate report identities, duplicate task IDs, or
  any report that fails the existing evidence identity chain.
- Let `series.to_evidence()` produce exact-circuit evidence whose evidence
  identity is the complete series identity. Grant no unseen-circuit estimate.

## Public API

```python
series = experiment.validation_series(
    [first_hardware_report, second_hardware_report],
    circuit=circuit,
    confidence_level=0.95,
)

print(series.mean_twin_qpu_agreement)
print(series.mean_ideal_qpu_agreement)
print(series.mean_qpu_repeatability)
print(series.simultaneous_finite_shot_tv_radius)

evidence = series.to_evidence()
fq.twin.dump_evidence(evidence, "twin-evidence.json")
```

Agreement is exactly `1 - total_variation_distance`; it is a comparison of
classical measurement-output distributions. It must not be labeled state
fidelity, amplitude accuracy, or per-shot success probability.

## Boundary

The series performs deterministic offline validation and aggregation. It does
not contact a provider, submit or poll hardware tasks, fetch calibration, train
or update a model, promote structural support, route workloads, or make an
application decision. A published envelope can be loaded by every user, but
publication, tenancy, and service distribution remain outside FlagQuantum.

## Compatibility

The change is additive. Existing single-report evidence, envelope persistence,
prediction, and evidence-report schemas remain unchanged. The candidate Twin
contract records the new type and method; the Stable Core root manifest does
not change.

## Acceptance

- One report produces the same conservative evidence radius as the existing
  single-report path.
- Two or more distinct reports expose prediction, ideal baseline, and observed
  repeatability separately.
- The final error bound covers the worst observed prediction error plus its
  simultaneous finite-shot radius, capped at one.
- Duplicated tasks and any changed identity or mapping fail closed.
- `to_evidence()` remains exact-circuit-only and round-trips through the
  existing `dump_evidence()` and `load_evidence()` functions.
- Documentation and the pull request contain complete executable API examples.
