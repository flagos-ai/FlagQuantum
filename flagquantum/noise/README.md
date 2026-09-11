# Noise Models and Calibration

This domain owns Kraus channel definitions, rules attaching channels to gates,
classical readout errors, device calibration specifications, and their serialized
identities. It depends on Core and Torch for tensor construction and validation.
It does not choose execution backends, schedule trajectories, submit jobs, or
implement state evolution.

Use `NoiseModel`, channel factories, and readout/calibration records from
`flagquantum.noise`. Compiler owns noise lowering; Runtime owns execution;
Simulation owns density-matrix and trajectory numerics. The compatibility
`noisy_density_matrix` export lazily delegates to Runtime and is not a domain
numerical implementation.

Start with classical readout to see the model's direction convention:

```python
import torch
from flagquantum.noise import NoiseModel, ReadoutError

# Rows are true states; columns are observed states.
readout = ReadoutError(((0.9, 0.1), (0.2, 0.8)))
model = NoiseModel().add_readout(0, readout)
observed = model.apply_readout_probabilities(
    torch.tensor([1.0, 0.0]), n_wires=1
)
torch.testing.assert_close(observed, torch.tensor([0.9, 0.1]))

restored = NoiseModel.from_dict(model.to_dict())
assert restored.identity == model.identity
```

For a channel change, edit `channels.py` and verify an analytic state or expectation
in `tests/test_noise.py`. For serialization changes, exercise both a valid round
trip and malformed inputs in `tests/unit/test_noise_deserialization.py`. Restoring
a model must enforce the same wire-count and probability constraints as direct
construction. Changes to serialized schemas or identity semantics require the
repository's contract review process.

`NoiseModel` is mutable; callers can add rules. Frozen channel/calibration records
do not make their nested Torch tensors immutable. Do not assume these objects
are safe shared mutable state across concurrent executions.
