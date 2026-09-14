# QPU digital twins

Use `fq.twin` to build a calibration-conditioned model of a mapped QPU, predict
its classical measurement distribution, and compare that prediction with
identity-bound hardware results.

## Offline prediction for any QPU

Start with a FlagQuantum `NoiseModel` that contains a `DeviceNoiseProfile`:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
twin = fq.twin.from_noise_model(
    device_noise_model,
    target="your-provider:your-qpu",
    qubits=(12, 13),
)
prediction = twin.predict(circuit)

print(prediction.twin_probabilities)
print(prediction.ideal_probabilities)
```

This path is offline. It does not contact a provider or submit a hardware task.
The logical wire order maps directly to the ordered physical `qubits` tuple.

## Native Quafu calibration

```python
from flagquantum.remote.qpu import QuafuProvider

provider = QuafuProvider()
chip_info = provider.fetch_chip_info("Shenglian")
twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Shenglian",
    qubits=(20, 27),
)
```

Fetching calibration performs provider I/O. Constructing the Twin from the
returned dictionary is local and does not schedule a QPU task.

## Save and restore the model

Persist the calibrated model before its source calibration changes:

```python
fq.twin.dump_twin(twin, "shenglian-twin.json")
```

Restore the same snapshot and model in a later process:

```python
import flagquantum as fq

restored = fq.twin.load_twin("shenglian-twin.json")
circuit = fq.Circuit(2).h(0).cx(0, 1)
prediction = restored.predict(circuit)

print(restored.snapshot.identity)
print(prediction.twin_probabilities)
```

Loading is entirely offline and reproduces the frozen model identity; it does
not fetch current calibration or contact a provider. The private mode-0600 file
contains the full provider-neutral noise model but no credential or task
receipt. It is a model artifact, not proof that the model is still accurate.

## Compare calibration drift

Compare two saved Twins for the same QPU mapping:

```python
import flagquantum as fq

reference = fq.twin.load_twin("shenglian-before.json")
current = fq.twin.load_twin("shenglian-after.json")
drift = fq.twin.compare_calibrations(reference, current)

print(f"max T1 change: {drift.maximum_relative_t1_change:.2%}")
print(f"max T2 change: {drift.maximum_relative_t2_change:.2%}")
if drift.maximum_readout_tv_distance is not None:
    print(f"max readout drift: {drift.maximum_readout_tv_distance:.2%}")
```

`qubit_drifts` identifies every result by physical qubit;
`gate_duration_drifts` identifies gate scopes by physical qubits, including
couplers. This gives topology and history applications stable facts to display.
The comparison normalizes timing units and requires the same target, mapping,
logical calibration structure, and gate scopes.

Relative changes are signed `(current - reference) / reference`; summary values
use absolute magnitudes. Readout drift is total-variation distance between
conditional measurement rows. `has_observed_drift` only says represented
calibration values differ—it does not prove statistical significance, reduced
prediction accuracy, or an expired trust window. Comparison is offline and does
not refresh or update either Twin.

## Build a calibration history

Load frozen models in strictly increasing calibration time, then build both a
cumulative series from the first snapshot and an incremental series between
adjacent snapshots:

```python
import flagquantum as fq

twins = [
    fq.twin.load_twin("shenglian-state-01.json"),
    fq.twin.load_twin("shenglian-state-02.json"),
    fq.twin.load_twin("shenglian-state-03.json"),
]
history = fq.twin.build_calibration_history(twins)
fq.twin.dump_calibration_history(history, "shenglian-history.json")

restored = fq.twin.load_calibration_history("shenglian-history.json")

for timestamp, cumulative, incremental in zip(
    restored.captured_at[1:],
    restored.baseline_drifts,
    restored.interval_drifts,
):
    print(
        timestamp,
        f"from baseline: {cumulative.maximum_relative_t1_change:.2%}",
        f"since previous: {incremental.maximum_relative_t1_change:.2%}",
    )
```

Every snapshot must identify the same provider, backend, ordered physical
mapping, and comparable calibration structure. Duplicate identities and
non-increasing timestamps fail closed. `history.to_dict()` is JSON-ready for a
drift chart, but FlagQuantum does not fetch snapshots on a schedule, persist a
history database, set trust thresholds, or update a model automatically.
The same workflow is available as a copy-and-run command:

```bash
python examples/twin_calibration_history.py \
  shenglian-state-01.json shenglian-state-02.json shenglian-state-03.json \
  --output shenglian-history.json
```

The canonical history file is written with private permissions. Repeating the
same write is idempotent; an existing different or invalid file is never
replaced. Loading validates every nested drift and derived summary and performs
no provider operation.

## Freeze a hardware validation

```python
experiment = fq.twin.TwinExperiment.prepare(
    twin,
    circuit,
    name="frozen-bell",
    shots=1024,
)
```

Preparation freezes the Twin snapshot and prediction, circuit identity,
canonical OpenQASM 2.0, target backend, ordered physical mapping, and shot count.
It does not submit a task. The explicit submission boundary is:

```python
receipt = experiment.submit(provider)  # submits one real Quafu task
```

Save the exact experiment and receipt together when the QPU may remain queued
after this process exits:

```python
submission = fq.twin.TwinSubmission.from_receipt(experiment, receipt)
fq.twin.dump_submission(submission, "twin-submission.json")
```

Resume in a later process without creating another task:

```python
from flagquantum.remote.qpu import QuafuProvider

provider = QuafuProvider()
submission = fq.twin.load_submission("twin-submission.json")
status = provider.query_status(submission.receipt)  # one explicit status query

if status == "Finished":
    result = provider.fetch_result(submission.receipt)
    hardware_report = submission.validate_result(result)
```

Loading is offline. It does not poll, submit, retry, cancel, or fetch anything.
The private mode-0600 file contains no credentials, is identity-bound to the
submitted QASM and ordered physical mapping, and is never overwritten with
different content.

Poll with the provider workflow, fetch the terminal result, and bind it back to
the exact receipt:

```python
result = provider.fetch_result(receipt)
hardware_report = experiment.validate_result(result, receipt=receipt)
```

For a complete program with bounded polling and two real repetitions, use
[`examples/remote/quafu_twin_evidence.py`](../../examples/remote/quafu_twin_evidence.py).
Running that file submits two 1,024-shot Shenglian tasks.

## Aggregate repeated hardware reports

Each report must come from a different task submitted from the same frozen
experiment:

```python
series = experiment.validation_series(
    [first_hardware_report, second_hardware_report],
    circuit=circuit,
    confidence_level=0.95,
)

print(f"Twin ↔ QPU: {series.mean_twin_qpu_agreement:.2%}")
print(f"Ideal SV ↔ QPU: {series.mean_ideal_qpu_agreement:.2%}")
print(f"QPU repeatability: {series.mean_qpu_repeatability:.2%}")
print(
    "Simultaneous finite-shot TV radius: "
    f"{series.simultaneous_finite_shot_tv_radius:.2%}"
)
```

These percentages compare classical measurement distributions using total
variation distance. They are not state fidelity, amplitude accuracy, or the
probability that one shot is correct.

## Save and restore validation

```python
fq.twin.dump_validation_series(series, "twin-validation.json")
restored_series = fq.twin.load_validation_series("twin-validation.json")

evidence = restored_series.to_evidence()
fq.twin.dump_evidence(evidence, "twin-evidence.json")
restored_evidence = fq.twin.load_evidence("twin-evidence.json")
report = twin.evidence_report(circuit, evidence=restored_evidence)

print(report.status)
print(report.tv_error_bound)
print(report.confidence_level)
```

Both loaders are offline and fail closed on malformed or unknown v1 fields.
Writers create canonical files and refuse to replace different evidence.

The validation-series artifact retains display metrics and repetition facts.
The evidence envelope is the reduced support object used to assess one
prediction. Neither artifact is a signed provider receipt or a raw-count
reproduction bundle.

## Track validation across calibrations

Pair each saved Twin with the validation series produced for that exact
snapshot. Every pair must use the same fixed circuit and ordered physical
mapping:

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
history = fq.twin.load_validation_history("shenglian-validation-history.json")

for timestamp, twin_match, ideal_match, qpu_repeatability, shot_radius, bound in zip(
    history.captured_at,
    history.mean_twin_qpu_agreements,
    history.mean_ideal_qpu_agreements,
    history.mean_qpu_repeatabilities,
    history.simultaneous_finite_shot_tv_radii,
    history.verified_tv_error_bounds,
    strict=True,
):
    print(timestamp, twin_match, ideal_match, qpu_repeatability, shot_radius, bound)
```

This is a longitudinal history of classical measurement-distribution metrics,
not state fidelity. It preserves Twin↔QPU agreement, noiseless Ideal↔QPU
agreement, observed QPU repeatability, and evidence-qualified TV bounds as
separate values. Snapshots must be chronological and unique; hardware-report
identities cannot be reused across observations.

Run the same offline workflow from the command line:

```bash
python examples/twin_validation_history.py \
  --observation shenglian-state-01.json shenglian-validation-01.json \
  --observation shenglian-state-02.json shenglian-validation-02.json \
  --output shenglian-validation-history.json
```

Building the history never fetches calibration, submits hardware work, updates
a model, or decides whether an application should trust or route a workload.
The optional output is canonical private JSON. Loading it verifies its strict
schema, nested series, all cross-observation invariants, and every derived
metric; tampered summaries and destructive replacement are rejected.

To incorporate one newly validated calibration, extend the saved history
immutably and write a new versioned artifact:

```python
import flagquantum as fq

history = fq.twin.load_validation_history("shenglian-history-state-02.json")
later_twin = fq.twin.load_twin("shenglian-state-03.json")
later_series = fq.twin.load_validation_series("shenglian-validation-03.json")

updated = history.append(later_twin, later_series)
fq.twin.dump_validation_history(updated, "shenglian-history-state-03.json")

print(updated.captured_at[-1])
print(updated.mean_twin_qpu_agreements[-1])
print(updated.verified_tv_error_bounds[-1])
```

The original object and file remain unchanged. The append fails if the new
observation is out of order, repeats a snapshot or hardware report, changes the
target, mapping, circuit identity or structure, or is not bound to the supplied
Twin. Scheduling, collecting, validating and approving new hardware evidence
remain explicit operations outside this method.

## Evidence statuses

- `exact_circuit_verified`: later hardware verified this exact circuit.
- `within_evidence_envelope`: the circuit is structurally in scope and an
  externally supplied estimated bound exists.
- `unverified`: a prediction exists, but no applicable empirical bound exists.
- `out_of_scope`: the supplied evidence belongs to another snapshot, mapping,
  circuit structure, or identity.

Evidence does not automatically authorize routing or hardware control. Global
publication, dashboards, access control, drift monitoring, agents, and MCP are
application responsibilities outside FlagQuantum.

## Version 1 compatibility

The public `fq.twin` v1 API, `flagquantum.qpu_digital_twin.v1`,
`flagquantum.twin_submission.v1`,
`flagquantum.twin_evidence_envelope.v1`, and
`flagquantum.twin_validation_series.v1` are frozen compatibility contracts.
Compatible capabilities may be added, but existing v1 names, signatures,
fields, status meanings, and serialized meanings will not change without a
versioned replacement or the documented deprecation process.
