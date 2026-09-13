# QPU digital twin

Connect a calibrated device model to a traceable hardware experiment. The
design direction is prediction, validation, and model refinement with explicit
uncertainty, physical mapping, and program identity.

Twin owns snapshots, mapped models, predictions, and comparison reports.
Noise owns channels, Simulation owns evolution, and Remote owns calibration
adapters and task submission.

## Compare a prediction

Given a Quafu calibration payload in `chip_info` and its physical-qubit mapping:

```python
import flagquantum as fq
from flagquantum.twin import QPUDigitalTwin

twin = QPUDigitalTwin.from_quafu_chip_info(
    chip_info,
    backend_name="Baihua",
    physical_qubits=(3, 4),
)
prediction = twin.predict(fq.Circuit(2).h(0).cx(0, 1))
report = prediction.compare_counts({"00": 500, "11": 500})
```

The counts above illustrate the comparison interface; they are not a live
hardware observation. Use `TwinExperiment` to bind a real submission and result.

## Decide whether a prediction is supported

`predict()` always returns the numerical result of the frozen model. It does not
claim that hardware has verified the prediction. Use `assess()` with a frozen
support envelope when software needs an explicit decision boundary:

```python
from flagquantum.twin import TwinSupportEnvelope

circuit = fq.Circuit(2).h(0).cx(0, 1)
support = TwinSupportEnvelope(
    snapshot_identity=twin.snapshot.identity,
    physical_qubits=(3, 4),
    supported_operations=("h", "cx"),
    maximum_instruction_count=2,
    verified_circuit_identities=(circuit.to_ir().content_hash,),
    evidence_identity=hardware_evidence_identity,
    verified_tv_error_radius=0.04,
    estimated_tv_error_radius=0.08,
    confidence_level=0.95,
)
assessment = twin.assess(circuit, support=support)
```

The decision is one of:

- `verified_prediction`: this exact circuit has a bound from later hardware;
- `bounded_estimate`: the circuit is inside a declared structural envelope and
  carries a separately established estimate bound;
- `physical_reference`: a model prediction exists, but no actionable evidence
  bound is available;
- `unsupported`: the snapshot, physical mapping, or circuit is outside the
  envelope.

`assessment.actionable` is true only for the two bounded decisions. The total
variation radius is a distribution-level error bound, not state fidelity or a
per-shot success probability. Constructing an envelope does not create evidence;
its identities and bounds must come from a frozen validation workflow.

## Verify a change

From the repository root:

```bash
python -m pytest \
  tests/test_twin.py \
  tests/test_twin_assessment.py \
  tests/test_twin_experiment.py -q
```

Check calibration identity, physical mapping, task binding, and rejection of
mismatched results. Distinguish prospective predictions from retrospective
diagnostics using evidence of the circuit actually executed.

[Experiment binding and validation scope](IMPLEMENTATION.md) describes the
Quafu lifecycle and the exact single-circuit density-matrix path. A calibration
model does not imply general hardware equivalence.
