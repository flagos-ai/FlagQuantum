# Native hybrid subset-majority QNN

Run from the repository root:

```bash
python -m examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.overview.main
```

Open `walkthrough.ipynb` with the repository on Python's import path. The full
workflow loads the reviewed checkpoint, trains 340 ten-bit inputs for one epoch
and tests 512 disjoint examples. Both quantum outputs are sampled with 4096 shots
per circuit execution, including shifted gradient evaluations.

```python
from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.overview.workflow import run_workflow

workflow = run_workflow(seed=17)
print([step.size for step in workflow.steps])  # [64, 64, 64, 64, 64, 20]
print(workflow.accuracy)
print(workflow.quantum_evaluations)  # 5612
```

The native qfunc quantum kernel remains in the parent directory. This directory
adds data preparation, portable checkpoint values, two classical linear layers,
explicit quantum parameter-shift gradients and actual sampled training/readout.
It does not introduce a general differentiable qfunc layer. See
[WORKFLOW.md](WORKFLOW.md) for source mapping, precision and sampling semantics.
