# QPU digital twin

Build calibration-conditioned digital models for arbitrary QPUs and connect
their predictions to traceable hardware evidence. The framework surface is
provider-neutral; provider integrations supply calibration and execution data.

Twin owns snapshots, mapped models, predictions, and comparison reports.
Noise owns channels, Simulation owns evolution, and Remote owns calibration
adapters and task submission.

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
# Poll explicitly through the provider, then fetch the matching result.
result = provider.fetch_result(handle)
hardware_report = experiment.validate_result(result, receipt=handle)
evidence = experiment.evidence_from_report(hardware_report, circuit=circuit)
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

The snippet above continues from the provider and Twin construction shown in
the preceding sections. For a complete copy-and-run Quafu workflow—including
token validation, live calibration retrieval, bounded polling, evidence
persistence, and report loading—run
[`examples/remote/quafu_twin_evidence.py`](../../examples/remote/quafu_twin_evidence.py).

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
  tests/test_twin_experiment.py -q
```

Check calibration identity, physical mapping, task binding, and rejection of
mismatched results. Distinguish prospective predictions from retrospective
diagnostics using evidence of the circuit actually executed.

[Experiment binding and validation scope](IMPLEMENTATION.md) describes the
Quafu lifecycle and the exact single-circuit density-matrix path. A calibration
model does not imply general hardware equivalence.
