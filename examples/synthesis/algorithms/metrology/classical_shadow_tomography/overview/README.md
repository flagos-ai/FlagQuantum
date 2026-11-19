# Native classical shadow tomography

Run from the repository root:

```bash
python -m examples.synthesis.algorithms.metrology.classical_shadow_tomography.overview.main
```

Open `walkthrough.ipynb` with the repository on Python's import path. The full
workflow collects 400 local Pauli and 50 depth-20 random Clifford snapshots,
with one actual shot per snapshot and the original seed 552.

```python
from examples.synthesis.algorithms.metrology.classical_shadow_tomography.overview.workflow import run_workflow

workflow = run_workflow()
print(len(workflow.pauli.bits), len(workflow.clifford.bits))  # 400 50
print(workflow.pauli.zz_estimate)  # 0.81
print(workflow.pauli_bound)  # (81065, 10), computed, not executed
```

This directory owns snapshot collection, matrix reconstruction, statistics and
teaching plots. It composes public native qfunc, execution and readout APIs;
it does not own a compiler, numerical backend or generic tomography service.
The parent directory's frozen quantum counterpart remains separate. Change the
workflow here and run `tests/integration/test_synthesis_shadow_notebook.py`.
See [WORKFLOW.md](WORKFLOW.md) for source mapping, bit order and finite-sample
limitations. No credentials or external quantum SDK are required.
