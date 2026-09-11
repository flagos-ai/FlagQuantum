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
- `TwinExperiment`: a prediction bound to the exact program submitted;
- `TwinHardwareReport`: a result bound to its experiment and remote task;
- `TwinSnapshot`: immutable calibration and model identity;
- `TwinPrediction`: ideal and calibration-conditioned probabilities;
- `TwinValidationReport`: comparison with a hardware observation.

## Ten-minute path

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

For hardware validation, freeze the prediction and submitted program before
submission. Remote polling remains the provider's responsibility:

```python
from flagquantum.twin import TwinExperiment

experiment = TwinExperiment.prepare(
    twin,
    circuit,
    submitted_qasm=submitted_qasm,
    name="frozen-bell",
    shots=1024,
)
handle = experiment.submit(provider)
# Poll through the provider, then fetch the matching result.
result = provider.fetch_result(handle)
hardware_report = experiment.validate_result(result, receipt=handle)
```

The public Quafu task path does not return an authoritative final circuit before
submission. Local QuarkCircuit or QSteed transpilation can produce a useful
candidate, but it is not a provider-issued execution receipt. Quafu may still
lower gates or remap qubits when compilation was not requested.

Consequently, `TwinHardwareReport.validation_scope` is
`"retrospective_diagnostic"`. The separately reported
`executed_program_matches_submission` flag records whether the final circuit
returned after execution has the same digest as the frozen submission; it does
not turn the result into a strict pre-execution circuit-level prediction.
Missing or rewritten executed programs fail closed.

Run the focused checks with:

```bash
python -m pytest tests/test_twin.py -q
python -m pytest tests/test_twin_experiment.py -q
```

The current implementation is an exact, single-circuit density-matrix path.
It is a calibration-driven emulator with explicit validation evidence, not a
claim of pulse-level or generally predictive hardware equivalence.
