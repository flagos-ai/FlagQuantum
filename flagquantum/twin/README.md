# QPU digital twin

`twin` owns calibration-conditioned, hardware-validated digital models of
quantum processing units. It freezes a mapped device calibration, predicts a
measurement distribution through FlagQuantum's noise and simulation paths, and
compares that prediction with later hardware counts.

It does not own numerical simulation, noise-channel definitions, provider
credentials, task submission, or raw vendor calibration parsing. Those remain
in `simulation`, `noise`, and `remote/qpu` respectively.

## Public entry points

- `QPUDigitalTwin`: a frozen device model bound to a physical mapping;
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

Run the focused checks with:

```bash
python -m pytest tests/test_twin.py -q
```

The current implementation is an exact, single-circuit density-matrix path.
It is a calibration-driven emulator with explicit validation evidence, not a
claim of pulse-level or generally predictive hardware equivalence.
