# Local basics

These are the shortest complete FlagQuantum workflows. They use the default
local CPU path, require no credentials, and make no remote or distributed
requests.

```bash
python -m examples.local.simulate
python -m examples.local.measure
python -m examples.local.train
```

- `simulate.py` builds a Bell circuit and checks its statevector.
- `measure.py` requests probabilities, a Pauli expectation, and sampled counts.
- `train.py` optimizes an `fq.Module` with a standard PyTorch optimizer.

After these pass, use the
[single-machine quantum AI examples](../single_machine_quantum_ai/README.md)
for configurable CPU/GPU, MPS, tensor-network, and hybrid-model workflows.
