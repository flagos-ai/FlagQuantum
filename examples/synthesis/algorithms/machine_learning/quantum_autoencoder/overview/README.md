# Native quantum autoencoder

Run from the repository root:

```bash
python -m examples.synthesis.algorithms.machine_learning.quantum_autoencoder.overview.main
```

Open `walkthrough.ipynb` with the repository on Python's import path. Both cost
routes execute one epoch from the same checkpoint, with three source inputs,
batch size two, 4096 shots, L1 loss and SGD learning rate 0.3.

```python
from examples.synthesis.algorithms.machine_learning.quantum_autoencoder.overview.workflow import run_workflow

workflow = run_workflow()
print([row.modal_bits for row in workflow.decoded])
# [(0, 0, 1, 1), (0, 0, 0, 1), (0, 1, 1, 1)]
print([row.data for row in workflow.anomalies if row.detected])
# [(1, 0, 1, 0), (1, 1, 1, 1)]
```

This directory owns the complete local teaching workflow and plots. It composes
public synthesis, execution, readout and PyTorch APIs; it does not own compiler,
runtime or general differentiable-layer implementations. The parent directory's
frozen quantum-counterpart example remains separate. Change this workflow here
and run `tests/integration/test_synthesis_autoencoder_notebook.py`.
See [WORKFLOW.md](WORKFLOW.md) for source and acceptance semantics.
