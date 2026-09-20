# Algorithm examples

Runnable demonstrations of the algorithm units in `flagquantum.algorithms`, one
per script. [`docs/guides/ALGORITHMS.md`](../../docs/guides/ALGORITHMS.md) is the
per-unit reference they follow, and the place each unit's advantage premise is
recorded in full.

Run them from the repository root:

```bash
python -m examples.algorithms.pca
python -m examples.algorithms.kmedians
python -m examples.algorithms.quantum_kernel
python -m examples.algorithms.feature_selection
python -m examples.algorithms.qarm
python -m examples.algorithms.svd
```

[`tests/test_algorithm_examples.py`](../../tests/test_algorithm_examples.py) runs
every script in this directory as a subprocess and asserts on what it printed, so
a script that stops working fails the test lane instead of going stale.

What they show:

- [`pca.py`](pca.py): the eigenvalue readout of a 2x2 data matrix's density
  matrix, by phase estimation over its exponential.
- [`kmedians.py`](kmedians.py): one k-medians assignment step, each point's
  nearest centroid found by a Grover-style minimum search, then the classical
  median update.
- [`quantum_kernel.py`](quantum_kernel.py): a kernel matrix estimated by swap
  test, and a classical kernel ridge classifier fitted on the estimated entries.
- [`feature_selection.py`](feature_selection.py): one feature-selection objective
  built and evaluated. It runs no circuit, because the unit has no quantum part.
- [`qarm.py`](qarm.py): the fraction of a database's items whose support meets a
  threshold, by amplitude estimation over a support register.
- [`svd.py`](svd.py): a matrix's singular values read off the phase of its
  Hermitian embedding's exponential, plus the boundary of a one-wire counting
  register.

## These scripts use the subpackage surface

The algorithm units are reachable at `flagquantum.algorithms.<unit>` and carry no
root-level `fq.` name. These scripts therefore import from the subpackage --
`from flagquantum.algorithms.pca import principal_components` -- rather than
through `import flagquantum as fq`, and `examples/README.md` records that
boundary.

Each script prints the premise its unit rests on, because the premise is the part
that is easiest to lose: quantum PCA's density matrix, its exponential and the
purification's amplitudes are all built classically here, quantum k-medians
compares a distance table built classically and synthesizes its oracle from a
truth table, quantum kernel estimation builds each feature state gate by gate
from a classical vector, feature selection runs no solver, the frequent-item
fractions iterate the transactions in Python, and the singular values come from a
state built out of the classical `torch.linalg.svd` the readout estimates. The
guide holds the full boundary for each.
