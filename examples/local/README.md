# Local basics

These are the shortest complete FlagQuantum workflows. They use the default
local CPU path, require no credentials, and make no remote or distributed
requests.

```bash
python -m examples.local.simulate
python -m examples.local.measure
python -m examples.local.train
python -m examples.local.dynamic_feedback
```

- `simulate.py` builds a Bell circuit and checks its statevector.
- `measure.py` requests probabilities, a Pauli expectation, and sampled counts.
- `train.py` optimizes an `fq.Module` with a standard PyTorch optimizer.
- `dynamic_feedback.py` uses the candidate-stable `DynamicCircuit` builder and
  experimental local executor for mid-circuit measurement, conditional feedback,
  reset, and projection to the canonical execution result.

After these pass, use the
[single-machine quantum AI examples](../single_machine_quantum_ai/README.md)
for configurable CPU/GPU, MPS, tensor-network, and hybrid-model workflows.
