# Twin convergence completion

## Outcome

The `twin_convergence` architecture migration is complete. The authoritative
framework implementation is `flagquantum/twin`; the former research worktree is
no longer an implementation authority for FlagQuantum.

This completion follows the explicit Twin v1 freeze recorded by API Change
Proposal 033. It closes a repository migration, not the scientific development
of digital twins.

## Completed framework path

The public, provider-neutral construction and prediction path is:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
twin = fq.twin.from_noise_model(
    device_noise_model,
    target="your-provider:your-qpu",
    qubits=(12, 13),
)
prediction = twin.predict(circuit)
```

Quafu is a native calibration adapter using the same target grammar:

```python
twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Shenglian",
    qubits=(20, 27),
)
```

Hardware validation remains explicit and identity bound:

```python
experiment = fq.twin.TwinExperiment.prepare(
    twin,
    circuit,
    name="frozen-bell",
    shots=1024,
)
receipt = experiment.submit(provider)  # submits one real hardware task
result = provider.fetch_result(receipt)
report = experiment.validate_result(result, receipt=receipt)
```

Preparing and predicting are offline. Only `submit()` creates a QPU task.

## Boundary proof

- Twin owns snapshots, predictions, validation reports, evidence, and repeated
  validation summaries.
- Noise owns calibration-conditioned channel semantics.
- Simulation owns numerical evolution.
- Remote owns provider credentials, calibration retrieval, task submission,
  polling, and result decoding.
- No Q-ATLAS stage name or research state machine is exported through
  `flagquantum.twin`.
- Agents, natural-language invocation, MCP, dashboards, global publication,
  routing policy, and automatic hardware control remain outside this framework
  namespace.

## What remains open

Migration completion does not claim pulse-level equivalence, arbitrary-circuit
accuracy, continuous self-evolution, or validity across calibration windows.
Those are future, additive capabilities and evidence programs. They must not
weaken or reinterpret the frozen Twin v1 contract.

The additive prospective candidate-trial API now supports evidence-qualified
comparison of two frozen models on one later result. It still does not automate
calibration collection, model construction, promotion, scheduling, or routing.
Candidate submissions can now be persisted as one strict binding containing the
complete trial and its existing provider receipt, then restored for result
validation without a resubmission path.

Candidate circuit suites now compare the same two frozen models over multiple
distinct circuits on one ordered physical mapping. They preserve per-task
identity binding and apply simultaneous finite-shot confidence correction, but
they do not generalize evidence beyond the predeclared suite or promote a model.
