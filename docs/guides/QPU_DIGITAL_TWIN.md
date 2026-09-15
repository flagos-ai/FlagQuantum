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

## Align device drift with Twin accuracy

Use the exact same snapshot sequence to place the calibration-drift and
validation timelines side by side:

```python
import flagquantum as fq

calibration = fq.twin.load_calibration_history("shenglian-calibration-history.json")
validation = fq.twin.load_validation_history("shenglian-validation-history.json")
evolution = fq.twin.align_histories(calibration, validation)

print(evolution.latest_calibration_drift)
print(evolution.latest_twin_agreement_change)
print(evolution.latest_ideal_agreement_change)
print(evolution.latest_qpu_repeatability_change)
print(evolution.latest_verified_bound_change)
```

The executable offline form is:

```bash
python examples/twin_evolution_history.py \
  --calibration-history shenglian-calibration-history.json \
  --validation-history shenglian-validation-history.json
```

`latest_calibration_drift` is the latest adjacent-interval drift record. The
other values are signed changes from the preceding observation: positive
agreement means closer measurement distributions, while a negative verified-
bound change means a tighter error bound. The full change sequences remain
available on the evolution object.

Alignment requires identical provider, backend, ordered physical mapping,
snapshot identities and capture times. It deliberately reports correlation in
time without claiming that calibration drift caused a prediction change. It
does not choose a threshold, retrain or promote a Twin, or submit hardware.

## Compare an incumbent and candidate prospectively

Freeze both model predictions before collecting one future QPU result:

```python
import flagquantum as fq

incumbent = fq.twin.load_twin("twin-incumbent.json")
candidate = fq.twin.from_quafu_chip_info(
    current_chip_info,
    target="quafu:Shenglian",
    qubits=incumbent.snapshot.physical_qubits,
)
circuit = fq.Circuit(2).h(0).cx(0, 1)
trial = fq.twin.prepare_candidate_trial(
    incumbent,
    candidate,
    circuit,
    name="candidate-bell",
    shots=1024,
)

# This is the workflow's only QPU submission.
receipt = trial.experiment.submit(provider)
result = provider.fetch_result(receipt)
evaluation = trial.validate_result(
    result,
    receipt=receipt,
    circuit=circuit,
    confidence_level=0.95,
)

print(evaluation.decision)
print(f"incumbent agreement: {1 - evaluation.incumbent_hardware_total_variation:.2%}")
print(f"candidate agreement: {1 - evaluation.candidate_hardware_total_variation:.2%}")
print(
    "candidate improvement interval:",
    evaluation.candidate_improvement_lower_bound,
    evaluation.candidate_improvement_upper_bound,
)
```

The candidate must be a later snapshot of the same target and ordered physical
mapping. Both predictions use one fixed circuit and are compared with the same
hardware counts, avoiding a second task as a source of QPU variation. The
reported improvement is incumbent TV distance minus candidate TV distance. Its
conservative uncertainty radius is twice the finite-shot TV radius because both
distances depend on the same empirical distribution.

`improved` and `degraded` require the confidence interval to exclude zero;
`inconclusive` means the collected shots cannot distinguish the models. This is
not state fidelity, a causal claim, or permission to replace the incumbent.

For a complete Quafu program with token checking and bounded polling, run:

```bash
python examples/remote/quafu_twin_candidate.py
```

That example loads `twin-incumbent.json`, fetches one newer Shenglian
calibration, and explicitly submits one 1,024-shot task.

### Resume a queued candidate comparison

Persist the complete comparison and its one existing provider receipt
immediately after explicit submission:

```python
receipt = trial.experiment.submit(provider)  # submits exactly one task
submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
fq.twin.dump_candidate_submission(
    submission,
    "twin-candidate-submission.json",
)
```

A later process can fetch and validate that same task without a submission
operation:

```python
submission = fq.twin.load_candidate_submission(
    "twin-candidate-submission.json"
)
result = provider.fetch_result(submission.receipt)
evaluation = submission.validate_result(result, circuit=circuit)
```

The restored object has no `submit()` method. Its private, create-once file
binds the complete incumbent and candidate trial to one allowlisted provider
receipt and rejects unknown fields, changed identities, mismatched experiments,
or replacement with different content.

For a complete two-command Quafu program, run:

```bash
python examples/remote/quafu_twin_candidate_resume.py submit
python examples/remote/quafu_twin_candidate_resume.py resume
```

The first command is the only command that creates hardware work. If the
provider accepted a task but the process exited before saving the checkpoint,
reconcile the printed task ID or provider task history; do not blindly rerun
`submit`.

## Compare a candidate across fixed circuits

A single trial proves only one exact circuit. Freeze a small, predeclared suite
when the question is whether a candidate is closer to later QPU measurements
across several circuit structures on the same physical mapping:

```python
import flagquantum as fq

incumbent = fq.twin.load_twin("twin-incumbent.json")
candidate = fq.twin.from_quafu_chip_info(
    current_chip_info,
    target="quafu:Shenglian",
    qubits=incumbent.snapshot.physical_qubits,
)
circuits = (
    fq.Circuit(2).h(0).cx(0, 1),
    fq.Circuit(2).x(0).cx(0, 1),
    fq.Circuit(2).h(1).cx(1, 0).x(1),
)
suite = fq.twin.prepare_candidate_suite(
    incumbent,
    candidate,
    circuits,
    name="candidate-workloads",
    shots=1024,
)
fq.twin.dump_candidate_suite(suite, "candidate-suite.json")
```

Submission remains explicit and individually checkpointed:

```python
submissions = []
for index, trial in enumerate(suite.trials, start=1):
    receipt = trial.experiment.submit(provider)  # one QPU task
    submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
    fq.twin.dump_candidate_submission(
        submission,
        f"candidate-submission-{index:02d}.json",
    )
    submissions.append(submission)
```

After the application has fetched one terminal result for every retained
receipt, validate the complete suite without another submission:

```python
evaluation = suite.validate_results(
    submissions,
    results,
    circuits=circuits,
    confidence_level=0.95,
)

print("decision:", evaluation.decision)
print(f"incumbent Twin ↔ QPU: {evaluation.mean_incumbent_qpu_agreement:.2%}")
print(f"candidate Twin ↔ QPU: {evaluation.mean_candidate_qpu_agreement:.2%}")
print(f"ideal SV ↔ QPU: {evaluation.mean_ideal_qpu_agreement:.2%}")
print(
    "95% mean improvement interval:",
    evaluation.candidate_improvement_lower_bound,
    evaluation.candidate_improvement_upper_bound,
)
```

The suite applies a simultaneous confidence correction over its fixed circuits.
It rejects missing, reordered, duplicate, or mismatched trials, submissions,
results, circuits, snapshots, mappings, programs, and task IDs. Its mean
agreement concerns classical measurement distributions for this suite only; it
is not quantum-state fidelity, arbitrary-circuit accuracy, or a promotion
decision.

The complete Quafu workflow deliberately separates offline preparation,
one indexed submission per command, and submission-free evaluation:

```bash
python examples/remote/quafu_twin_candidate_suite.py prepare
python examples/remote/quafu_twin_candidate_suite.py submit 1
python examples/remote/quafu_twin_candidate_suite.py submit 2
python examples/remote/quafu_twin_candidate_suite.py submit 3
python examples/remote/quafu_twin_candidate_suite.py evaluate
```

## Evidence statuses

### Qualify evidence by physical topology and depth

`TwinEvidenceEnvelope` records statistical evidence and its basic structural
scope. Use `TwinCircuitSupport` when the validation also predeclared physical
couplers and a circuit-depth limit:

```python
support = fq.twin.TwinCircuitSupport(
    evidence=evidence,
    directed_couplers=((20, 27), (27, 20), (27, 34), (34, 27)),
    maximum_circuit_depth=8,
)

report = support.evidence_report(twin, circuit)
if report.status == "out_of_scope":
    print(report.reasons)

fq.twin.dump_circuit_support(support, "twin-circuit-support.json")
support = fq.twin.load_circuit_support("twin-circuit-support.json")
```

Couplers are directed because an executed two-qubit gate may not have equivalent
evidence in the reverse direction. Single-qubit operations need no coupler.
Operations on more than two wires are outside the first contract. Circuit depth
is computed from instruction dependencies on logical wires, then the ordered
Twin mapping translates every two-wire instruction to physical qubits.

This object can only narrow the supplied evidence. It does not infer support
from provider topology, compose independently validated cells, estimate a new
bound, or claim arbitrary-circuit accuracy. The complete offline example is
`examples/twin_circuit_support.py`.

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
`flagquantum.twin_candidate_submission.v1`,
`flagquantum.twin_candidate_suite.v1`,
`flagquantum.twin_candidate_suite_evaluation.v1`,
`flagquantum.twin_evidence_envelope.v1`, and
`flagquantum.twin_validation_series.v1` are frozen compatibility contracts.
Compatible capabilities may be added, but existing v1 names, signatures,
fields, status meanings, and serialized meanings will not change without a
versioned replacement or the documented deprecation process.
