# Native QML guide

Run from the repository root:

```bash
python -m examples.synthesis.tutorials.basics.qml.overview.main
```

Open `walkthrough.ipynb` with the repository on Python's import path. Both sections
use native Python qfunc definitions and actual local quantum execution.

```python
from examples.synthesis.tutorials.basics.qml.overview.workflow import run_workflow

workflow = run_workflow()
print(workflow.variational.optimizer.value)
print(workflow.training.predictions, workflow.training.accuracy)
```

The default local VQE should approach energy -0.61803398875; NOT training keeps
five MAE/SGD updates from weight 3. Accuracy is evaluated on the same two
NOT examples; it is not a generalization result. Report actual convergence and accuracy.
Exact native readout, native Nelder-Mead and explicit parameter-shift weight
gradients adapt the original sampled VQE/QLayer workflow. A generic differentiable
qfunc layer is not added. See [WORKFLOW.md](WORKFLOW.md) for the source mapping,
training contract and independent acceptance.
