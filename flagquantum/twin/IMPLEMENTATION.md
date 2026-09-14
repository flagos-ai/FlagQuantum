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
