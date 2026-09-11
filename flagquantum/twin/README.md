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

## Verify a change

From the repository root:

```bash
python -m pytest tests/test_twin.py tests/test_twin_experiment.py -q
```

Check calibration identity, physical mapping, task binding, and rejection of
mismatched results. Distinguish prospective predictions from retrospective
diagnostics using evidence of the circuit actually executed.

[Experiment binding and validation scope](IMPLEMENTATION.md) describes the
Quafu lifecycle and the exact single-circuit density-matrix path. A calibration
model does not imply general hardware equivalence.
