# QPU digital twin

Build calibration-conditioned digital models for arbitrary QPUs and connect
their predictions to traceable hardware evidence. The framework surface is
provider-neutral; provider integrations supply calibration and execution data.

Twin owns snapshots, mapped models, predictions, and comparison reports.
Noise owns channels, Simulation owns evolution, and Remote owns calibration
adapters and task submission.

See the [QPU digital-twin guide](../../docs/guides/QPU_DIGITAL_TWIN.md) for the
complete offline, Quafu validation, repetition, persistence, and interpretation
workflows.

## Build a Twin for any QPU

Start from a FlagQuantum `NoiseModel` carrying a device profile. The execution
target and ordered physical mapping become part of the immutable Twin identity.
The target uses the same `provider:backend` form as `fq.run`:

```python
import flagquantum as fq

twin = fq.twin.from_noise_model(
    device_noise_model,
    target="your-provider:your-qpu",
    qubits=(12, 13),
)
prediction = twin.predict(fq.Circuit(2).h(0).cx(0, 1))
```

Any QPU integration can construct the same framework model after converting its
calibration into FlagQuantum's provider-neutral noise and device-profile types.

## Native Quafu support

Quafu calibration conversion is built in, so no custom adapter is needed:

```python
twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Baihua",
    qubits=(3, 4),
)
prediction = twin.predict(fq.Circuit(2).h(0).cx(0, 1))
report = prediction.compare_counts({"00": 500, "11": 500})
```

The counts above illustrate the comparison interface; they are not a live
hardware observation. Use `TwinExperiment` to bind a real submission and result.

Persist the complete model when predictions must remain reproducible across
processes or calibration changes:

```python
fq.twin.dump_twin(twin, "qpu-twin.json")
restored = fq.twin.load_twin("qpu-twin.json")
prediction = restored.predict(fq.Circuit(2).h(0).cx(0, 1))
```

The strict v1 model artifact contains the frozen snapshot and provider-neutral
noise model, but no credentials or task receipt. Loading is offline and does not
refresh calibration, submit work, or claim that the saved model remains current.

Compare two model snapshots for the same physical mapping without provider I/O:

```python
before = fq.twin.load_twin("qpu-before.json")
after = fq.twin.load_twin("qpu-after.json")
drift = fq.twin.compare_calibrations(before, after)

print(drift.maximum_relative_t1_change)
print(drift.maximum_readout_tv_distance)
for qubit in drift.qubit_drifts:
    print(qubit.physical_qubit, qubit.relative_t2_delta)
```

The report uses physical identifiers for qubits and gate scopes so an
application can draw topology overlays. It reports calibration changes, not
Twin-to-QPU accuracy, statistical significance, or a policy decision.

Build chart-ready history from saved snapshots without provider I/O:

```python
twins = [
    fq.twin.load_twin("qpu-state-01.json"),
    fq.twin.load_twin("qpu-state-02.json"),
    fq.twin.load_twin("qpu-state-03.json"),
]
history = fq.twin.build_calibration_history(twins)
fq.twin.dump_calibration_history(history, "qpu-history.json")
history = fq.twin.load_calibration_history("qpu-history.json")

for timestamp, cumulative, incremental in zip(
    history.captured_at[1:], history.baseline_drifts, history.interval_drifts
):
    print(timestamp, cumulative.maximum_relative_t1_change)
    print(timestamp, incremental.maximum_relative_t1_change)
```

The cumulative series compares every later snapshot with the first; the
incremental series compares adjacent snapshots. The builder requires one
strictly chronological, identity-unique series for the same target and ordered
mapping. It does not collect calibrations, manage a database, or decide when a
Twin is trustworthy.

The history artifact is canonical, credential-free, and written once with
private permissions. Loading validates its nested drift records and derived
summaries without fetching calibration or restoring the source Twin models.

The shortest identity-bound validation path lets FlagQuantum emit the submitted
OpenQASM directly from the circuit. OpenQASM emission adds measurement of every
circuit qubit, so no measurement operation is added to the `Circuit` itself:

```python
experiment = fq.twin.TwinExperiment.prepare(
    twin,
    circuit,
    name="frozen-bell",
    shots=1024,
)
handle = experiment.submit(provider)
submission = fq.twin.TwinSubmission.from_receipt(experiment, handle)
fq.twin.dump_submission(submission, "twin-submission.json")

# A later process restores the same task; loading never submits or polls.
submission = fq.twin.load_submission("twin-submission.json")
# Poll explicitly through the provider, then fetch the matching result.
result = provider.fetch_result(submission.receipt)
hardware_report = submission.validate_result(result)
evidence = submission.experiment.evidence_from_report(
    hardware_report,
    circuit=circuit,
)
fq.twin.dump_evidence(evidence, "twin-evidence.json")
```

`evidence_from_report()` verifies the circuit, canonical OpenQASM, receipt,
result, authoritative executed program, physical mapping, counts, and shot
count before producing exact-circuit evidence. Quafu may lower gates after
submission; that provider-attested transformation is accepted only when the
result echoes the frozen source, uses exactly the selected physical qubits, and
preserves their measurement order. This is not an independent proof of compiler
semantic equivalence or state fidelity. The TV radius adds a conservative
multinomial finite-shot radius to the observed Twin-to-hardware distance. It
grants no estimate for unseen circuits.

The submission artifact is credential-free, written with private mode-0600
permissions, and never replaces different or invalid content. Its strict v1
loader restores only the frozen experiment and original task receipt; provider
construction, status queries, result retrieval, cancellation, and retry policy
remain explicit application responsibilities.

The snippet above continues from the provider and Twin construction shown in
the preceding sections. For a complete copy-and-run Quafu workflow—including
token validation, live calibration retrieval, bounded polling, evidence
persistence, and report loading—run
[`examples/remote/quafu_twin_evidence.py`](../../examples/remote/quafu_twin_evidence.py).

## Summarize repeated validation

Keep Twin error, the noiseless baseline, QPU repeatability, and shot uncertainty
separate when the same frozen experiment is executed more than once:

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

Track the same fixed validation circuit across later frozen calibrations:

```python
observations = [
    (
        fq.twin.load_twin("qpu-state-01.json"),
        fq.twin.load_validation_series("validation-state-01.json"),
    ),
    (
        fq.twin.load_twin("qpu-state-02.json"),
        fq.twin.load_validation_series("validation-state-02.json"),
    ),
]
history = fq.twin.build_validation_history(observations)
fq.twin.dump_validation_history(history, "validation-history.json")
history = fq.twin.load_validation_history("validation-history.json")

print(history.mean_twin_qpu_agreements)
print(history.mean_ideal_qpu_agreements)
print(history.mean_qpu_repeatabilities)
print(history.simultaneous_finite_shot_tv_radii)
print(history.verified_tv_error_bounds)
```

The builder verifies snapshot identity, target, ordered physical mapping,
circuit identity and structure, chronological order, and globally distinct
hardware reports. These values remain measurement-distribution comparisons;
they are not state fidelity or a workload-routing decision.

The canonical history file is private, credential-free, and safe to load
offline. Saving the same content is idempotent; different or invalid existing
content is never replaced. Loading recomputes every derived metric from the
nested validation series and rejects tampering.

Append a later validated calibration without rebuilding the earlier history:

```python
history = fq.twin.load_validation_history("validation-history-state-02.json")
later_twin = fq.twin.load_twin("qpu-state-03.json")
later_series = fq.twin.load_validation_series("validation-state-03.json")

updated = history.append(later_twin, later_series)
fq.twin.dump_validation_history(updated, "validation-history-state-03.json")
```

`append()` returns a new immutable history. It requires a later unique snapshot,
the same target, physical mapping, fixed circuit and structure, and previously
unused hardware reports. Use a new output path so earlier evidence remains
append-only and auditable.

Align device drift with validation changes only when both histories contain the
exact same snapshots:

```python
calibration = fq.twin.load_calibration_history("calibration-history.json")
validation = fq.twin.load_validation_history("validation-history.json")
evolution = fq.twin.align_histories(calibration, validation)

print(evolution.latest_calibration_drift)
print(evolution.latest_twin_agreement_change)
print(evolution.latest_ideal_agreement_change)
print(evolution.latest_qpu_repeatability_change)
print(evolution.latest_verified_bound_change)
```

These are synchronized observations, not a causal model: the API never claims
that a measured calibration change caused an accuracy change.

Each agreement is `1 - TV distance` for classical measurement-output
distributions. QPU repeatability is the pairwise agreement between observed
hardware distributions and still includes finite-shot noise. The verified
bound uses the worst distinct execution and a simultaneous confidence
correction; repetitions do not establish support for a different circuit.

Persist the complete validation summary when a downstream product needs the
separate display metrics after the process exits:

```python
fq.twin.dump_validation_series(series, "twin-validation.json")
restored_series = fq.twin.load_validation_series("twin-validation.json")

print(f"Twin ↔ QPU: {restored_series.mean_twin_qpu_agreement:.2%}")
print(f"Ideal SV ↔ QPU: {restored_series.mean_ideal_qpu_agreement:.2%}")
```

The writer creates a canonical file, permits an idempotent save of the same
series, and refuses to replace different or invalid content. The artifact is
an offline summary rather than a signed provider receipt. FlagQuantum does not
publish it or assign application access policy.

## Inspect evidence for a prediction

`predict()` always returns the numerical result of the frozen model. It does not
claim that hardware has verified the prediction. Use `evidence_report()` to
inspect the empirical support for that result:

```python
circuit = fq.Circuit(2).h(0).cx(0, 1)
evidence = fq.twin.TwinEvidenceEnvelope(
    snapshot_identity=twin.snapshot.identity,
    physical_qubits=(3, 4),
    supported_operations=("h", "cx"),
    maximum_instruction_count=2,
    verified_circuit_identities=(circuit.to_ir().content_hash,),
    evidence_identity=hardware_evidence_identity,
    verified_tv_error_bound=0.04,
    estimated_tv_error_bound=0.08,
    confidence_level=0.95,
)
report = twin.evidence_report(circuit, evidence=evidence)
```

Production evidence can be written and restored without provider access:

```python
fq.twin.dump_evidence(evidence, "twin-evidence.json")
evidence = fq.twin.load_evidence("twin-evidence.json")
report = twin.evidence_report(circuit, evidence=evidence)

print(report.status)
print(report.tv_error_bound)
print(report.confidence_level)
```

`dump_evidence()` writes canonical JSON once. Repeating it with the same
evidence is safe; it refuses to replace a different or invalid file.

The loader accepts only the complete `flagquantum.twin_evidence_envelope.v1`
schema and fails on missing or unknown fields. Research artifacts must first be
converted by their owning validation workflow. A converter may bind evidence
only when the frozen Twin snapshot identity and FlagQuantum IR circuit
identities were recorded before the target hardware outcomes. Physical-QASM
hashes, calibration identifiers, or retrospective matches cannot substitute
for those identities.

Q-ATLAS research workflows can use
`tools/convert_q_atlas_twin_evidence.py` after freezing the companion identity
binding. The tool is an offline compatibility boundary, not a public framework
API. Existing historical audits that did not record this bridge prospectively
remain useful research evidence but cannot be relabeled as evidence for a
different `QPUDigitalTwin` snapshot.

The evidence status is one of:

- `exact_circuit_verified`: this exact circuit has a bound from later hardware;
- `within_evidence_envelope`: the circuit is inside a declared structural scope and
  carries a separately established estimate bound;
- `unverified`: a model prediction exists, but no empirical bound is available;
- `out_of_scope`: the snapshot, physical mapping, or circuit is outside the
  envelope.

The report contains model and evidence facts, not an execution or routing
decision. Applications may apply their own policies outside FlagQuantum. The
total-variation bound applies to measured output distributions; it is not state
fidelity or a per-shot success probability. Constructing an envelope does not
create evidence. Its identities and bounds must come from a frozen validation
workflow.

## Verify a change

From the repository root:

```bash
python -m pytest \
  tests/test_twin.py \
  tests/test_twin_evidence.py \
  tests/test_twin_experiment.py \
  tests/test_twin_validation_series.py -q
```

Check calibration identity, physical mapping, task binding, and rejection of
mismatched results. Distinguish prospective predictions from retrospective
diagnostics using evidence of the circuit actually executed.

[Experiment binding and validation scope](IMPLEMENTATION.md) describes the
Quafu lifecycle and the exact single-circuit density-matrix path. A calibration
model does not imply general hardware equivalence.
