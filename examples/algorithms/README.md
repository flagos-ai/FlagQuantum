# Algorithm examples

Runnable demonstrations of the Phase 2 algorithm units in
`flagquantum.algorithms`, one per unit of that phase.
[`docs/guides/ALGORITHMS.md`](../../docs/guides/ALGORITHMS.md) is the per-unit
reference they follow, and the place each unit's advantage premise is recorded in
full.

Run them from the repository root:

```bash
python -m examples.algorithms.pca
python -m examples.algorithms.kmedians
python -m examples.algorithms.quantum_kernel
python -m examples.algorithms.feature_selection
python -m examples.algorithms.qarm
python -m examples.algorithms.svd
python -m examples.algorithms.error_mitigation
python -m examples.algorithms.pec
python -m examples.algorithms.spsa_optimizer
python -m examples.algorithms.trotter
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
- [`error_mitigation.py`](error_mitigation.py): an observable continued to zero
  noise by polynomial least squares and by Richardson extrapolation over four
  scaled models, with the residual of an underfit, of a square fit, and of a
  non-polynomial family printed beside the estimate.
- [`pec.py`](pec.py): a declared Pauli channel inverted from its Pauli transfer
  matrix and inserted after the channel it inverts, with the cost that composes
  over locations printed beside the estimate, and the four channel families whose
  transfer matrix is not diagonal refused by name together with a channel whose
  transfer eigenvalue reaches zero.
- [`spsa_optimizer.py`](spsa_optimizer.py): a Pauli energy minimized from samples
  at two evaluations per step, with the parameter-shift gradient's own evaluation
  count measured beside it.
- [`trotter.py`](trotter.py): a transverse-field Ising Hamiltonian turned into the
  circuit a product formula applies, with the defect at two step counts per order
  measured against `torch.matrix_exp`, the primitive's own emitted gates printed
  for six words, and the resulting circuit run through the static resource
  estimator and the gradient path.

## These scripts use the subpackage surface

The algorithm units are reachable at `flagquantum.algorithms.<unit>` and carry no
root-level `fq.` name, with one exception. The algorithm scripts import from the
subpackage -- `from flagquantum.algorithms.pca import principal_components` --
rather than through `import flagquantum as fq`. `spsa_optimizer.py` imports both:
the optimizer from the subpackage, and `flagquantum` itself for the `fq.Circuit`
and `fq.run` calls its objective makes. `trotter.py` needs no root alias either:
the circuit it builds is a `flagquantum.circuit.Circuit`, and the readout it takes
is the package's own `expectation_ps`. `examples/README.md` records that
boundary.

Each script prints the premise its unit rests on, because the premise is the part
that is easiest to lose: quantum PCA's density matrix, its exponential and the
purification's amplitudes are all built classically here, quantum k-medians
compares a distance table built classically and synthesizes its oracle from a
truth table, quantum kernel estimation builds each feature state gate by gate
from a classical vector, feature selection runs no solver, the frequent-item
fractions iterate the transactions in Python, the singular values come from a
state built out of the classical `torch.linalg.svd` the readout estimates,
zero-noise extrapolation rests on a polynomial-in-the-scale-factor assumption
that is not checkable from the measurements it fits, probabilistic error
cancellation rests on the noise being exactly the channel the model declares at
the location it declares it and pays for the inversion in programs rather than
shots, the SPSA update is built from a finite-difference estimate that is an
estimate rather than a gradient, and the product formula approximates the
evolution with a defect that is measured rather than bounded, because a bound
needs a commutator norm the caller has to supply. The guide holds the full
boundary for each.
