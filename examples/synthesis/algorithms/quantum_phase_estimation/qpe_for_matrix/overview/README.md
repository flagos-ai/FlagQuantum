# Matrix eigenvalues with native QPE

Run the complete local workflow:

```sh
python -m examples.synthesis.algorithms.quantum_phase_estimation.qpe_for_matrix.overview.main
```

Open `walkthrough.ipynb` for the two sampled distributions and eigenvalue plot.
NumPy and Matplotlib are optional example dependencies. The reusable workflow
uses FlagQuantum `qfunc` for preparation, exact evolution and power-dependent
second-order Suzuki evolution. Each route uses the original seven signed phase
bits and 10000 shots. No source importer, account or remote execution is used.

```python
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_for_matrix.overview.workflow import run_estimate

estimate = run_estimate("approximate", seed=1401)
print(estimate.solutions)
print(estimate.intervals)
print(estimate.resources)
```

The source's approximation can produce a broad distribution; its reported
intervals are the original display convention, not a new confidence guarantee.
The two dropped data bits remain physically entangled. Zero clean workspace
therefore means nine total physical bits, not seven.

See [workflow and acceptance](WORKFLOW.md), [provenance](provenance.json) and the
[third-party license](../../../../CLASSIQ_LICENSE.txt).
