# API Change Proposal 100: Immutable Twin candidate fitting

## Status

**Implementation authorized; additive to the Twin framework API.** This slice
turns one bounded `local_pauli_readout` estimate into a later immutable
candidate model. It never collects calibration, submits hardware, or releases
the candidate.

## Problem

`TwinParameterEstimate` deliberately contains observations and uncertainty but
no model-mutation authority. Applications nevertheless need a reproducible way
to turn that result into a candidate that existing prospective holdout workflows
can evaluate. Ad hoc conversion would lose the probe-bank identity, obscure
which values were fitted versus retained, and risk changing the incumbent in
place.

## Decision

- Add `fq.twin.fit_candidate(incumbent, estimate, *, captured_at)`.
- Return `TwinFittedCandidate`, a `QPUDigitalTwin` subtype accepted by existing
  prediction, candidate-suite, and holdout APIs.
- Record `TwinCandidateFit` provenance: incumbent snapshot, estimate, probe
  bank, parameter family, fitted values and intervals, retained calibration
  fields, and explicitly unidentified parameters.
- Bind the provenance identity into the candidate calibration source, and
  therefore into both noise-model and snapshot identities.
- Persist fitted candidates as `flagquantum.qpu_digital_twin.v2`; continue to
  read and write ordinary version-1 Twin files unchanged.

## Public API

The complete offline executable is
[`examples/twin_candidate_fit.py`](../../examples/twin_candidate_fit.py):

```bash
python examples/twin_candidate_fit.py
```

The application call after a successful bounded estimate is:

```python
candidate = fq.twin.fit_candidate(
    incumbent,
    estimate,
    captured_at="2026-08-15T10:30:00+08:00",
)

assert candidate.snapshot.identity != incumbent.snapshot.identity
print(candidate.fit.fitted_parameters)
print(candidate.fit.retained_parameters)
print(candidate.fit.parameter_estimate_identity)

fq.twin.dump_twin(candidate, "twin-candidate.json")
restored = fq.twin.load_twin("twin-candidate.json")
assert restored.snapshot.identity == candidate.snapshot.identity
```

## Numerical interpretation

The estimator's values are computational-basis total-variation burdens within
the declared first-order local model. Candidate fitting maps readout burden to
a symmetric bit-flip matrix. It maps one-qubit burden `b` to depolarizing
probability `3b/2` and two-qubit burden `b` to `5b/4`, because those channels
produce computational-basis TV burdens `2p/3` and `4p/5` respectively.

Values above the representable ranges fail closed. The conversion is not
arbitrary channel tomography, process fidelity, pulse reconstruction, or a
claim that the model generalizes outside its declared probes and later holdout
tests.

## Identity and retained state

The estimate must bind the exact incumbent snapshot, canonical target, and
ordered physical mapping. The candidate retains T1, T2, excited population,
gate durations, target, and mapping from the incumbent while replacing local
readout and Pauli burdens with the fit. A later timezone-aware `captured_at` is
mandatory.

The incumbent object and its serialized noise model remain unchanged. The
candidate has a distinct calibration, noise-model, and snapshot identity even
when every fitted burden is zero.

## Fail-closed behavior

Fitting rejects a changed target, mapping, source snapshot, parameter family,
parameter-name set, non-increasing or timezone-naive timestamp, missing gate
scope, and a burden that cannot be represented by the declared depolarizing
channel. Loading rejects noncanonical provenance or a model whose calibration
does not bind that provenance.

## Acceptance

- Fitted, retained, and unidentified names are separately inspectable.
- Candidate provenance binds both probe-bank and estimate identities.
- `dump_twin`/`load_twin` reproduce candidate identity and predictions.
- The incumbent remains byte-for-byte unchanged.
- Existing candidate-suite and holdout preparation accepts the fitted Twin.
- No provider, task, receipt, submission, retry, routing, promotion, or release
  operation is introduced.

## Non-goals

- No hardware collection or calibration refresh.
- No arbitrary Kraus, leakage, crosstalk, pulse, or temporal model fitting.
- No automatic candidate selection, release, routing, or hardware control.
- No Agent, MCP, natural-language, REST, UI, or application policy.
