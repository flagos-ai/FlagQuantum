# Native teleportation Notebook

Run from the repository root:

```bash
python -m examples.synthesis.algorithms.foundational.quantum_teleportation.overview.main
```

Open `walkthrough.ipynb` with the repository on Python's import path. The user
prepares a logical input through two angles, runs the native qfunc, and reads
Bob's density matrix and observations without physical wire decoding.

```python
from examples.synthesis.algorithms.foundational.quantum_teleportation.overview.workflow import run_workflow

workflow = run_workflow(theta=0.73, phi=-0.41)
print(workflow.density)
print(dict(workflow.observations))
```

The original plus-state example is the default. Measurement is explicitly
deferred: Bob's output channel matches private measured feedback, but Alice's
outcomes remain an environment. This is not generic midcircuit measurement,
reset, or a simulation of physical classical transmission. The source leaves
its sample count unspecified; 1000 local observations and seed 3 are explicit
native choices. See [WORKFLOW.md](WORKFLOW.md) for acceptance and boundaries.
