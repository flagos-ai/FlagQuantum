# QPU digital twin

`twin` owns calibration-conditioned device models and comparisons with hardware
observations. Hardware validation is specific to an experiment and its evidence. It freezes a mapped device calibration, predicts a
measurement distribution through FlagQuantum's noise and simulation paths, and
compares that prediction with later hardware counts.

It does not own numerical simulation, noise-channel definitions, provider
credentials, task submission, or raw vendor calibration parsing. Those remain
in `simulation`, `noise`, and `remote/qpu` respectively.

## Public entry points

- `QPUDigitalTwin`: a frozen device model bound to a physical mapping;
- `TwinCalibrationDrift`: offline change facts between comparable Twin snapshots;
- `TwinCalibrationHistory`: cumulative and adjacent drift across chronological
  comparable Twin snapshots;
- `TwinExperiment`: a prediction bound to the exact program submitted;
- `TwinHardwareReport`: a result bound to its experiment and remote task;
- `TwinSubmission`: a persistable binding of an experiment to its original task;
- `TwinSnapshot`: immutable calibration and model identity;
- `TwinPrediction`: ideal and calibration-conditioned probabilities;
- `TwinValidationReport`: comparison with one hardware observation;
- `TwinValidationHistory`: fixed-circuit validation across chronological Twins;
- `TwinValidationSeries`: conservative summary of distinct repeated results.
- `TwinRegionHoldoutStudy`: a prospectively frozen reference/holdout circuit
  split for one regional Twin;
- `TwinRegionHoldoutEvaluation`: simultaneous evidence for both groups;
- `TwinRegionValidationHistory`: one fixed regional circuit suite across
  chronological calibration snapshots.
- `TwinRegionHoldoutHistory`: one fixed prospective reference/holdout design
  across chronological calibration snapshots, preserving separate holdout
  agreement and uncertainty trends without updating or promoting a model.
- `TwinRegionHoldoutEvolution`: exact snapshot alignment between regional
  holdout changes and authoritative calibration-drift intervals, without a
  causal or promotion claim.
- `TwinRegionCandidateHoldoutStudy`: a frozen regional incumbent/candidate
  comparison with disjoint reference and holdout circuits;
- `TwinRegionCandidateHoldoutEvaluation`: one simultaneous decision in which
  holdout improvement is required and reference degradation vetoes an upgrade.
- `TwinRegionRelease`: an explicit, immutable release manifest limited to the
  exact circuits in one improved regional candidate holdout evaluation.

## Ten-minute path

```python
import flagquantum as fq

twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Baihua",
    qubits=(3, 4),
)
prediction = twin.predict(fq.Circuit(2).h(0).cx(0, 1))
report = prediction.compare_counts({"00": 500, "11": 500})
```

Persist and restore that exact model without provider access:

```python
fq.twin.dump_twin(twin, "qpu-twin.json")
restored = fq.twin.load_twin("qpu-twin.json")
assert restored.snapshot.identity == twin.snapshot.identity
```

The model artifact is separate from evidence and submission artifacts. It
contains the full provider-neutral noise specification but no credentials,
remote task, accuracy proof, or automatic refresh behavior.

Compare two frozen models only when they identify the same target and mapping:

```python
drift = fq.twin.compare_calibrations(reference_twin, current_twin)
for qubit in drift.qubit_drifts:
    print(qubit.physical_qubit, qubit.relative_t1_delta)
```

Time values are normalized to seconds, and gate scopes are translated from
logical wires back to physical qubits. The report provides no significance
threshold and does not infer accuracy decay or update either model.

Build a pure, chart-ready history from already persisted Twins:

```python
history = fq.twin.build_calibration_history(
    [reference_twin, next_twin, current_twin]
)
fq.twin.dump_calibration_history(history, "calibration-history.json")
history = fq.twin.load_calibration_history("calibration-history.json")
for drift in history.baseline_drifts:
    print(drift.maximum_relative_t1_change)
for drift in history.interval_drifts:
    print(drift.maximum_relative_t1_change)
```

The first sequence is cumulative from `reference_twin`; the second is
incremental between neighbors. Input order and identity are validated, and no
provider operation or model mutation occurs.

The history persistence functions use canonical JSON and mode-0600 creation.
They permit an identical repeated write but refuse to replace different or
invalid content. Loading reconstructs and validates every nested public drift
record; it does not reconstruct the source models or contact a provider.

For hardware validation, freeze the prediction and submitted program before
submission. Remote polling remains the provider's responsibility:

```python
from flagquantum.twin import TwinExperiment

experiment = TwinExperiment.prepare(
    twin,
    circuit,
    name="frozen-bell",
    shots=1024,
)
handle = experiment.submit(provider)
submission = fq.twin.TwinSubmission.from_receipt(experiment, handle)
fq.twin.dump_submission(submission, "twin-submission.json")
# Poll through the provider, then fetch the matching result.
submission = fq.twin.load_submission("twin-submission.json")
result = provider.fetch_result(submission.receipt)
hardware_report = submission.validate_result(result)
evidence = submission.experiment.evidence_from_report(
    hardware_report,
    circuit=circuit,
)
```

The submission file allows a later process to continue the original task. It
contains no credentials and loading it performs no provider operation.

For repeated executions of that same frozen experiment, keep model error,
ideal-baseline error, observed QPU repeatability, and finite-shot uncertainty
separate:

```python
series = experiment.validation_series(
    [first_hardware_report, second_hardware_report],
    circuit=circuit,
)
evidence = series.to_evidence()
```

The series uses a simultaneous confidence correction and the worst distinct
execution for its verified bound. Pairwise QPU repeatability still contains
shot noise. It does not promote an unseen circuit or assume that hardware is
stationary across observations.

Persist the series separately from its reduced evidence envelope when an
application needs to retain its display and audit metrics:

```python
fq.twin.dump_validation_series(series, "twin-validation.json")
restored_series = fq.twin.load_validation_series("twin-validation.json")
```

The canonical JSON round-trip preserves the series identity. Existing
different or invalid files are never overwritten. Loading is offline and does
not contact a provider; publication and product history remain outside this
module.

Align persisted models and validation series to inspect accuracy evidence over
calibration time:

```python
history = fq.twin.build_validation_history(
    [
        (reference_twin, reference_series),
        (current_twin, current_series),
    ]
)
fq.twin.dump_validation_history(history, "validation-history.json")
history = fq.twin.load_validation_history("validation-history.json")
print(history.mean_twin_qpu_agreements)
print(history.mean_ideal_qpu_agreements)
print(history.mean_qpu_repeatabilities)
print(history.simultaneous_finite_shot_tv_radii)
print(history.verified_tv_error_bounds)
```

The paired input makes snapshot binding explicit. All observations must share
one QPU mapping and fixed circuit, appear in strictly increasing calibration
time, and use globally distinct hardware reports. Construction is offline and
does not assign a trust threshold or mutate a Twin.

Validation-history persistence uses a strict canonical schema. The loader
reconstructs the nested validation series and recomputes derived agreement and
uncertainty arrays; the private writer is idempotent only for identical content
and never replaces a different or malformed artifact.

Incremental evolution is immutable and append-only:

```python
history = fq.twin.load_validation_history("validation-history-state-02.json")
later_twin = fq.twin.load_twin("qpu-state-03.json")
later_series = fq.twin.load_validation_series("validation-state-03.json")

updated = history.append(later_twin, later_series)
fq.twin.dump_validation_history(updated, "validation-history-state-03.json")
```

The method reruns the complete history invariants and never mutates the model,
the previous history, or its persisted artifact. The framework does not decide
when to collect the next observation.

Align repeated evaluations of the same fixed regional circuit suite:

```python
region_history = fq.twin.build_region_validation_history(
    [
        (reference_region_twin, reference_evaluation),
        (current_region_twin, current_evaluation),
    ]
)
fq.twin.dump_region_validation_history(
    region_history,
    "region-validation-history.json",
)
region_history = fq.twin.load_region_validation_history(
    "region-validation-history.json"
)
print(region_history.mean_twin_qpu_agreements)
print(region_history.simultaneous_tv_error_bounds)
```

The target, mapping, full topology, ordered circuit suite, exercised couplers,
and maximum circuit depth remain fixed while calibration snapshots advance in
strict time order. Hardware reports cannot be reused across observations.
This is an offline observational record, not arbitrary-circuit evidence, a
trust window, a model update, or a scheduling policy.

Prospectively separate reference and holdout regional circuits before any QPU
work:

```python
study = fq.twin.prepare_region_holdout_study(
    region_twin,
    reference_circuits,
    holdout_circuits,
    physical_qubits=(20, 27, 34),
    name="regional-holdout",
    shots=1024,
    repetitions=2,
)
fq.twin.dump_region_holdout_study(study, "region-holdout-study.json")
study = fq.twin.load_region_holdout_study("region-holdout-study.json")

# After explicit submission, terminal-result fetching, and validation:
fq.twin.dump_region_holdout_evaluation(
    evaluation,
    "region-holdout-evaluation.json",
)
evaluation = fq.twin.load_region_holdout_evaluation(
    "region-holdout-evaluation.json",
)
```

The two groups share one snapshot, mapping, shot count, and repetition count,
while circuit identities and later QPU task identities must be disjoint. One
Bonferroni allocation covers both groups, every circuit, and every repetition.
The resulting holdout TV-error increase is an observation for the frozen
circuits, not an arbitrary-circuit, training, trust, or routing claim.
Evaluation loading reconstructs its nested suite and validation-series records
and verifies every serialized derived metric. Evaluation checkpoints use the
same private, create-once behavior as the frozen study.

Compare a later regional candidate on future hardware without duplicating QPU
tasks for the incumbent:

```python
candidate_study = fq.twin.prepare_region_candidate_holdout(
    incumbent_region_twin,
    candidate_region_twin,
    reference_circuits,
    holdout_circuits,
    physical_qubits=(20, 27, 34),
    name="regional-candidate",
    shots=1024,
)
fq.twin.dump_region_candidate_holdout_study(
    candidate_study,
    "region-candidate-holdout.json",
)

# Each trial's candidate-bound experiment is submitted once. That result is
# compared with both predictions by the existing TwinCandidateSubmission.
evaluation = candidate_study.validate_results(
    reference_submissions,
    reference_results,
    holdout_submissions,
    holdout_results,
    reference_circuits=reference_circuits,
    holdout_circuits=holdout_circuits,
)
fq.twin.dump_region_candidate_holdout_evaluation(
    evaluation,
    "region-candidate-evaluation.json",
)
evaluation = fq.twin.load_region_candidate_holdout_evaluation(
    "region-candidate-evaluation.json",
)
print(evaluation.decision)
```

Both regional models must have exactly the same target, ordered mapping,
directed topology, operations, and structural limits. `improved` requires a
statistically positive holdout result and no reference degradation. This is an
evidence gate only; applications retain model promotion and routing authority.
The evaluation checkpoint is private and create-once. Loading it reconstructs
all nested evidence records and rejects any noncanonical or modified field.

Explicitly freeze an improved candidate as an exact-circuit release:

```python
release = fq.twin.release_region_candidate(
    incumbent_region_twin,
    candidate_region_twin,
    study=candidate_study,
    evaluation=evaluation,
)
fq.twin.dump_region_release(release, "region-release.json")
release = fq.twin.load_region_release("region-release.json")

assert release.scope == "exact_circuits"
assert release.routing_authorized is False
print(release.candidate_region_identity)
print(release.verified_circuit_identities)
```

This is an application-triggered qualification artifact, not an automatic
model replacement. It performs no provider operation and grants no routing or
arbitrary-circuit authority.

Ask whether one exact circuit is covered by that release:

```python
release = fq.twin.load_region_release("region-release.json")
assessment = release.assess(
    candidate_region_twin,
    fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
    physical_qubits=(20, 27, 34),
)

print(assessment.status)
print(assessment.prediction)
print(assessment.reasons)
print(assessment.release_identity)
```

The status is `released_exact_circuit` only when the release identity, the
candidate regional model identity, the candidate snapshot, the provider/backend
target, the ordered physical mapping, the regional structural coverage, and the
frozen content hash of that exact circuit all match. That status returns a
`TwinPrediction` from the released model; `outside_release` returns no
prediction plus deterministic reason tokens, including the regional coverage
tokens. Every identity is derived internally, so no fingerprint is a user
input. An assessment claims no per-circuit confidence level and no
total-variation error bound from aggregate candidate-improvement evidence, and
it performs no provider operation, routing, or mutation of the release or the
model.

Align the two independently audited timelines by exact snapshot identity:

```python
calibration = fq.twin.load_calibration_history("calibration-history.json")
validation = fq.twin.load_validation_history("validation-history.json")
evolution = fq.twin.align_histories(calibration, validation)

print(evolution.latest_calibration_drift)
print(evolution.latest_twin_agreement_change)
print(evolution.latest_verified_bound_change)
```

`TwinEvolutionHistory` retains both source histories and derives signed adjacent
changes without a causal or promotion claim. Target, ordered mapping, snapshot
identities and captured times must match exactly.

Omitting `submitted_qasm` creates a direct, deterministic binding from the
FlagQuantum IR circuit to OpenQASM 2.0. Only that canonical form can be promoted
from a matching later hardware report into exact-circuit evidence. Supplying a
custom QASM program remains supported for retrospective diagnostics, but it is
rejected by `evidence_from_report()` because FlagQuantum cannot infer semantic
equivalence from unrelated text.

The public Quafu task path does not return an authoritative final circuit before
submission. Local QuarkCircuit or QSteed transpilation can produce a useful
candidate, but it is not a provider-issued execution receipt. Quafu may still
lower gates after submission even when service compilation was not requested.

`TwinHardwareReport.validation_scope` remains `"retrospective_diagnostic"`.
The separately reported
`executed_program_matches_submission` flag records whether the final circuit
returned after execution has the same digest as the frozen submission; it does
not by itself turn the result into verified evidence. Exact evidence additionally
requires the canonical FlagQuantum IR-to-QASM binding checked by
`evidence_from_report()`. A provider-transpiled program is accepted only when
the result echoes the exact frozen source program, its authoritative
`transpiled` field uses exactly the selected physical qubits, and measurements
preserve the ordered logical-to-physical mapping. Missing or custom programs,
extra physical qubits, and reordered measurements fail closed. The Quafu result
attests this lowering; FlagQuantum does not independently prove arbitrary
compiler semantic equivalence or state fidelity.

Run the focused checks with:

```bash
python -m pytest tests/test_twin.py -q
python -m pytest tests/test_twin_experiment.py -q
python -m pytest tests/test_twin_validation_series.py -q
```

The current implementation is an exact, single-circuit density-matrix path.
It is a calibration-driven emulator with explicit validation evidence, not a
claim of pulse-level or generally predictive hardware equivalence.
