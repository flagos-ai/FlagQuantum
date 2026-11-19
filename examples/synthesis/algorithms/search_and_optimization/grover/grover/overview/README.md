# Adaptive Grover search

Run the original three-part search workflow in native FlagQuantum:

```bash
python -m examples.synthesis.algorithms.search_and_optimization.grover.grover.overview.main
```

Open `walkthrough.ipynb` for both SAT truth tables, finite-shot search tables,
local circuit drawings, resource reports and the planar MaxCut partition plot.
The six code cells use the typed `workflow.py` helpers; no original Qmod or
external execution service is needed. Install the repository's optional notebook
plotting dependencies before running the graphical workflow.

```python
from examples.synthesis.algorithms.search_and_optimization.grover.grover.overview.workflow import run_search

search = run_search("small_3sat", seed=1100)
print(search.succeeded)
print(search.attempts[-1].outcomes[:3])
```

Each attempt executes the native quantum program and draws 1000 samples.
The loop stops only when one marked observed word strictly exceeds its
probability threshold. An exhausted loop retains its final attempt and reports
failure. The example deliberately preserves this distinction.

`workflow.py` owns classical search control, sampling, tables and graph display.
The existing native four-qubit SAT and five-qubit graph quantum functions are
reused; the three-qubit predicate is implemented as a native `qfunc` here.
Dependencies remain public FlagQuantum APIs plus Torch, Matplotlib and NetworkX.
It does not own a compiler, sampler backend or generic search framework.

Run `tests/integration/test_synthesis_grover_notebook.py` for independent Circuit
full-state and finite-shot references. Read `WORKFLOW.md` for source mapping,
bit order, stopping semantics and local acceptance scope.
