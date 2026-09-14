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
