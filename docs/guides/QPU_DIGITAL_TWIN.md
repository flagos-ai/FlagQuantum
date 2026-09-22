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

`evidence_report()` predicts the full computational-basis (Z-basis) output
distribution; it does not perform a live measurement. During prospective
validation, `TwinExperiment.prepare()` emits OpenQASM that measures every
mapped qubit in the same basis. Measurement is therefore bound in the hardware
experiment rather than stored as a terminal operation on `Circuit`.

Couplers are directed because an executed two-qubit gate may not have equivalent
evidence in the reverse direction. Single-qubit operations need no coupler.
Operations on more than two wires are outside the first contract. Circuit depth
is computed from instruction dependencies on logical wires, then the ordered
Twin mapping translates every two-wire instruction to physical qubits.

This object can only narrow the supplied evidence. It does not infer support
from provider topology, compose independently validated cells, estimate a new
bound, or claim arbitrary-circuit accuracy. The complete offline example is
`examples/twin_circuit_support.py`.

### Compose connected structural coverage

Several overlapping `TwinCircuitSupport` cells from the same QPU and calibration
capture can be combined into a structural region:

```python
region = fq.twin.compose_connected_region(
    [(twin_a, support_a), (twin_b, support_b)]
)

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
coverage = region.coverage_report(
    circuit,
    physical_qubits=(20, 27, 34),
)

print(coverage.status)
print(coverage.missing_qubits)
print(coverage.missing_directed_couplers)
```

The mapping is mandatory and ordered: logical wire `i` maps to
`physical_qubits[i]`. Composition rejects different targets, calibration
capture times, mismatched evidence identities, and disconnected cells. It uses
the most conservative common operation, instruction-count, and depth boundary.

Region coverage is structural only. Local cell error bounds are neither
averaged nor combined, so `coverage.tv_error_bound` and
`coverage.confidence_level` are always `None`. A region-level accuracy claim
requires prospective evidence for the complete mapped workload. See
`examples/twin_connected_region.py` for the complete offline workflow.

### Compose a runnable regional Twin model

When the local cells also carry compatible calibration and noise models, compose
them into one wider offline Twin:

```python
region_twin = fq.twin.compose_region_twin(
    [(twin_a, support_a), (twin_b, support_b)]
)

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
prediction = region_twin.predict(
    circuit,
    physical_qubits=(20, 27, 34),
)

print(prediction.twin_probabilities)
print(prediction.total_variation_from_ideal)
```

The mapping must exactly equal the model's canonical regional wire order.
Composition fails closed if overlapping cells disagree on qubit calibration,
gate duration, gate-noise channels, or readout noise. An unscoped gate-noise
rule is accepted only when every cell declares the same channel. No correlated
noise is inferred between cells.

`total_variation_from_ideal` compares the regional Twin's predicted Z-basis
measurement distribution with noiseless statevector simulation. It is not a
Twin-to-QPU accuracy measurement. A newly composed model therefore has no
regional accuracy bound or confidence level. The full offline code is
`examples/twin_region_model.py`.

### Prospectively validate one regional circuit

Freeze the complete mapped circuit before any hardware task is created:

```python
experiment = region_twin.prepare_experiment(
    circuit,
    physical_qubits=(20, 27, 34),
    name="regional-ghz",
    shots=1024,
)
```

`prepare_experiment()` checks the region mapping, directed couplers, operations,
instruction count, and depth, then binds the composed prediction to canonical
OpenQASM 2.0. It performs no provider I/O. Submission remains explicit through
`experiment.submit(provider)`.

After at least two distinct tasks have been validated with
`experiment.validate_result(...)`, create a repeated validation series and
qualify the exact circuit:

```python
series = experiment.validation_series(
    (first_report, second_report),
    circuit=circuit,
)
support = region_twin.support_from_validation_series(
    series,
    circuit,
    physical_qubits=(20, 27, 34),
)
report = support.evidence_report(region_twin.twin, circuit)

print(report.status)
print(series.mean_twin_qpu_agreement)
print(series.mean_ideal_qpu_agreement)
print(series.mean_qpu_repeatability)
print(report.tv_error_bound)
print(report.confidence_level)
```

The returned support contains only the directed couplers exercised by this
exact circuit. It does not validate arbitrary circuits on the region or
unexercised links. The three agreement values compare classical Z-basis
measurement distributions; they are not quantum-state fidelity. See
`examples/remote/quafu_twin_region_validation.py` for the complete token,
polling, repeated-submission, and persistence workflow.

### Validate a fixed regional circuit suite

One exact circuit does not establish agreement for other regional workloads.
Freeze at least two distinct covered circuits and at least two repetitions per
circuit before creating any QPU task:

```python
suite = fq.twin.prepare_region_validation_suite(
    region_twin,
    circuits,
    physical_qubits=(20, 27, 34),
    name="shenglian-region-suite",
    shots=1024,
    repetitions=2,
)

print(suite.planned_task_count)
print(suite.planned_shots)
```

Preparation is offline. The suite deliberately has no bulk submit method:
applications explicitly submit each `suite.experiments` member once per
repetition and persist the resulting `TwinSubmission`. This makes partial
network failures recoverable without silently resubmitting completed work.

After all planned tasks reach terminal results, validate the complete ordered
collection at once:

```python
evaluation = suite.validate_results(
    submissions,
    results,
    circuits=circuits,
    confidence_level=0.95,
)
support = evaluation.to_circuit_support()

print(evaluation.mean_twin_qpu_agreement)
print(evaluation.mean_ideal_qpu_agreement)
print(evaluation.mean_qpu_repeatability)
print(evaluation.simultaneous_finite_shot_tv_radius)
print(evaluation.simultaneous_tv_error_bound)
```

Confidence is allocated across every circuit and repetition. The returned
support lists every exact suite circuit as verified and only the directed
couplers exercised by those circuits. A different circuit remains unverified,
even if it uses the same qubits and operations. See
`examples/remote/quafu_twin_region_validation_suite.py` for the checkpointed
`prepare`, single-task `submit`, and all-results `evaluate` commands.

### Validate prospectively held-out regional circuits

A reference suite does not establish agreement for circuits chosen after its
results are known. Freeze a disjoint holdout group at the same time as the
reference group, before creating any QPU task:

```python
study = fq.twin.prepare_region_holdout_study(
    region_twin,
    reference_circuits,
    holdout_circuits,
    physical_qubits=(20, 27, 34),
    name="shenglian-region-holdout",
    shots=1024,
    repetitions=2,
)

print(study.reference_circuit_count)
print(study.holdout_circuit_count)
print(study.planned_task_count)
print(study.planned_shots)

fq.twin.dump_region_holdout_study(study, "region-holdout-study.json")
study = fq.twin.load_region_holdout_study("region-holdout-study.json")
```

The study deliberately has no bulk submit method. Submit and checkpoint every
experiment from `study.reference_suite` and `study.holdout_suite` explicitly.
After all predeclared tasks are terminal, validate both groups together:

```python
evaluation = study.validate_results(
    reference_submissions,
    reference_results,
    holdout_submissions,
    holdout_results,
    reference_circuits=reference_circuits,
    holdout_circuits=holdout_circuits,
    confidence_level=0.95,
)

fq.twin.dump_region_holdout_evaluation(
    evaluation,
    "region-holdout-evaluation.json",
)
evaluation = fq.twin.load_region_holdout_evaluation(
    "region-holdout-evaluation.json",
)

print(evaluation.reference_twin_qpu_agreement)
print(evaluation.holdout_twin_qpu_agreement)
print(evaluation.holdout_twin_qpu_tv_increase)
print(evaluation.holdout_ideal_qpu_agreement)
print(evaluation.holdout_qpu_repeatability)
print(evaluation.holdout_simultaneous_tv_error_bound)
```

`holdout_twin_qpu_tv_increase` is holdout mean Twin-QPU TV error minus
reference mean Twin-QPU TV error; positive values mean the holdout circuits
were harder for the Twin. Confidence is simultaneous across both groups,
their circuits, and every repetition. The result supports only the exact
predeclared circuits. It is not arbitrary-circuit accuracy, state fidelity, a
training-generalization claim, or permission to route production workloads.
Loading the evaluation reconstructs both nested suite evaluations and every
validation series, then rejects any changed derived metric or non-canonical
version-1 payload. Files are private, create-once checkpoints; writing the
identical evaluation again is idempotent.
The complete checkpointed workflow is
`examples/remote/quafu_twin_region_holdout.py`.

### Gate a later regional candidate on held-out circuits

Use future QPU results to compare a later regional candidate with the current
regional Twin. One candidate-bound task is submitted per circuit; the same
result is compared with both predictions, so the incumbent does not consume a
second hardware task.

```python
import flagquantum as fq


def compose_region(cell_artifacts):
    return fq.twin.compose_region_twin(
        [
            (
                fq.twin.load_twin(twin_path),
                fq.twin.load_circuit_support(support_path),
            )
            for twin_path, support_path in cell_artifacts
        ]
    )


incumbent_region = compose_region(
    [
        ("incumbent-q20-q27-twin.json", "incumbent-q20-q27-support.json"),
        ("incumbent-q27-q34-twin.json", "incumbent-q27-q34-support.json"),
    ]
)
candidate_region = compose_region(
    [
        ("candidate-q20-q27-twin.json", "candidate-q20-q27-support.json"),
        ("candidate-q27-q34-twin.json", "candidate-q27-q34-support.json"),
    ]
)

reference_circuits = (
    fq.Circuit(3).h(0).cx(0, 1).cx(1, 2),
    fq.Circuit(3).x(0).cx(0, 1),
)
holdout_circuits = (
    fq.Circuit(3).h(1).cx(1, 2),
    fq.Circuit(3).h(0).cx(0, 1).cx(1, 2).x(2),
)

study = fq.twin.prepare_region_candidate_holdout(
    incumbent_region,
    candidate_region,
    reference_circuits,
    holdout_circuits,
    physical_qubits=(20, 27, 34),
    name="shenglian-region-candidate",
    shots=1024,
)
fq.twin.dump_region_candidate_holdout_study(
    study,
    "region-candidate-holdout.json",
)

# Applications explicitly call trial.experiment.submit(provider), checkpoint
# each TwinCandidateSubmission, and fetch each terminal result. They then pass
# those ordered records back to the frozen study:
evaluation = study.validate_results(
    reference_submissions,
    reference_results,
    holdout_submissions,
    holdout_results,
    reference_circuits=reference_circuits,
    holdout_circuits=holdout_circuits,
    confidence_level=0.95,
)
fq.twin.dump_region_candidate_holdout_evaluation(
    evaluation,
    "region-candidate-evaluation.json",
)

# A later process can restore the complete, evidence-qualified decision.
evaluation = fq.twin.load_region_candidate_holdout_evaluation(
    "region-candidate-evaluation.json",
)

print(evaluation.decision)
print(evaluation.reference_evaluation.mean_candidate_improvement)
print(evaluation.holdout_evaluation.mean_candidate_improvement)
print(evaluation.holdout_evaluation.candidate_improvement_lower_bound)
```

The regional target, ordered mapping, complete directed topology, operations,
and structural limits must remain identical. The candidate snapshot must be
later, and reference/holdout circuits and task identities must be disjoint.
Confidence covers both groups and every circuit. `improved` requires holdout
improvement while reference degradation vetoes the upgrade; `degraded` or
`inconclusive` never replaces a model. The framework reports this decision but
does not submit automatically, retry, promote a candidate, or route workloads.
Evaluation checkpoints are private create-once files. Loading reconstructs the
nested suite, candidate, hardware, and validation records, recomputes every
derived metric, and rejects missing, extra, modified, or noncanonical data.

### Track holdout agreement across calibration snapshots

After the same predeclared reference and holdout circuits have been evaluated
at two or more calibration snapshots, build one offline longitudinal record:

```python
import flagquantum as fq


def load_region_twin(cell_artifacts):
    return fq.twin.compose_region_twin(
        [
            (
                fq.twin.load_twin(twin_path),
                fq.twin.load_circuit_support(support_path),
            )
            for twin_path, support_path in cell_artifacts
        ]
    )


state_01_twin = load_region_twin(
    [
        ("state-01-q20-q27-twin.json", "state-01-q20-q27-support.json"),
        ("state-01-q27-q34-twin.json", "state-01-q27-q34-support.json"),
    ]
)
state_02_twin = load_region_twin(
    [
        ("state-02-q20-q27-twin.json", "state-02-q20-q27-support.json"),
        ("state-02-q27-q34-twin.json", "state-02-q27-q34-support.json"),
    ]
)

state_01_evaluation = fq.twin.load_region_holdout_evaluation(
    "state-01-holdout-evaluation.json"
)
state_02_evaluation = fq.twin.load_region_holdout_evaluation(
    "state-02-holdout-evaluation.json"
)

history = fq.twin.build_region_holdout_history(
    [
        (state_01_twin, state_01_evaluation),
        (state_02_twin, state_02_evaluation),
    ]
)

print(history.reference_twin_qpu_agreements)
print(history.holdout_twin_qpu_agreements)
print(history.holdout_tv_error_increases)
print(history.holdout_simultaneous_tv_error_bounds)

fq.twin.dump_region_holdout_history(history, "region-holdout-history.json")
restored = fq.twin.load_region_holdout_history(
    "region-holdout-history.json"
)
```

Every observation must retain the same provider/backend, ordered physical
mapping, full directed topology, ordered reference and holdout circuit groups,
repetitions, shots, confidence, exercised couplers, and depth limits. Capture
times must strictly increase; snapshots, regional models, studies, and hardware
reports cannot be reused.

The history compares measurement-distribution agreement for the fixed design.
It does not retrain the Twin, infer a validity duration, prove arbitrary-circuit
accuracy, schedule QPU work, promote a model, or route workloads.

### Align calibration drift with holdout agreement

Load the two persisted histories and align them by exact snapshot identity and
capture time:

```python
import flagquantum as fq

calibration_history = fq.twin.load_calibration_history(
    "region-calibration-history.json"
)
holdout_history = fq.twin.load_region_holdout_history(
    "region-holdout-history.json"
)

evolution = fq.twin.align_region_holdout_history(
    calibration_history,
    holdout_history,
)

for values in zip(
    evolution.interval_maximum_relative_t1_changes,
    evolution.interval_maximum_relative_t2_changes,
    evolution.interval_maximum_readout_tv_distances,
    evolution.interval_maximum_relative_gate_duration_changes,
    evolution.holdout_twin_qpu_agreement_changes,
    evolution.holdout_simultaneous_tv_error_bound_changes,
    strict=True,
):
    print(values)
```

Every tuple has one value per adjacent calibration interval, making the payload
directly usable by a drift chart. A lower holdout-agreement change or higher
error-bound change is an observation, not proof that calibration drift caused
the change. The alignment does not choose thresholds, refresh a model, promote
a candidate, or route workloads.

### Track one regional suite across calibration snapshots

After the same frozen circuit suite has been evaluated against distinct later
hardware tasks at two or more calibration snapshots, align those evaluations
into one offline history:

```python
history = fq.twin.build_region_validation_history(
    [
        (reference_region_twin, reference_evaluation),
        (current_region_twin, current_evaluation),
    ]
)

print(history.mean_twin_qpu_agreements)
print(history.mean_ideal_qpu_agreements)
print(history.mean_qpu_repeatabilities)
print(history.simultaneous_finite_shot_tv_radii)
print(history.simultaneous_tv_error_bounds)
print(history.task_counts)
print(history.total_shots)

fq.twin.dump_region_validation_history(
    history,
    "twin-region-validation-history.json",
)
restored = fq.twin.load_region_validation_history(
    "twin-region-validation-history.json"
)

updated = restored.append(later_region_twin, later_evaluation)
```

The target, ordered physical mapping, full directed topology, ordered fixed
circuit identities, exercised couplers, and maximum circuit depth must remain
the same. Calibration and regional-model identities change at each strictly
later observation, and every hardware report may appear only once.

Construction, loading, and appending are offline. The history does not submit
or poll hardware, update the Twin, infer a trust window, or promote the model.
Agreement and bounds apply to classical measurement distributions for the
fixed suite only; they are neither quantum-state fidelity nor evidence for an
arbitrary circuit on the same region.

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
`flagquantum.twin_validation_series.v1`,
`flagquantum.twin_region_holdout_study.v1`, and
`flagquantum.twin_region_holdout_evaluation.v1`, and
`flagquantum.twin_region_holdout_history.v1`, and
`flagquantum.twin_region_holdout_evolution.v1`,
`flagquantum.twin_region_candidate_holdout_study.v1`, and
`flagquantum.twin_region_candidate_holdout_evaluation.v1` are frozen compatibility
contracts.
Compatible capabilities may be added, but existing v1 names, signatures,
fields, status meanings, and serialized meanings will not change without a
versioned replacement or the documented deprecation process.
