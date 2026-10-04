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
python -m examples.algorithms.block_encoding
python -m examples.algorithms.logical_resources
python -m examples.algorithms.arithmetic
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
- [`block_encoding.py`](block_encoding.py): a Hermitian matrix encoded into the
  flagged block of a unitary, with the block read back out of the circuit one
  basis state at a time and checked against both the matrix over `alpha` and the
  matrix itself, the walk step's eigenphases matched against the matrix's own
  spectrum from a different routine, a second implementation that holds no matrix
  read by the same consumer, and the ten construction and three register refusals
  printed by name.
- [`logical_resources.py`](logical_resources.py): a Clifford+T program costed on a
  rotated surface code, with the compiler's own resource record printed beside the
  report's copy of it so a second T rule would show, the distance sweep tabulated
  against the model's closed form, the one-layer charge for a measurement record
  shown against the same program without one, and the two refused opcode families,
  a lowered channel, and six refused arguments each printed by name.
- [`arithmetic.py`](arithmetic.py): a reversible adder built at seven widths and
  then run on the statevector path, with the sum decoded back out of the runtime's
  own state vector, the two ancillas' behaviour off the contract measured over
  every input rather than described, the T-cost read from the compiler's own
  Toffoli rule, and the conversion bound shown to be a bound the caller can raise.

## These scripts use the subpackage surface

The algorithm units are reachable at `flagquantum.algorithms.<unit>` and carry no
root-level `fq.` name, with one exception. The algorithm scripts import from the
subpackage -- `from flagquantum.algorithms.pca import principal_components` --
rather than through `import flagquantum as fq`. `spsa_optimizer.py` imports both:
the optimizer from the subpackage, and `flagquantum` itself for the `fq.Circuit`
and `fq.run` calls its objective makes. `trotter.py` needs no root alias either:
the circuit it builds is a `flagquantum.circuit.Circuit`, and the readout it takes
is the package's own `expectation_ps`. `block_encoding.py` needs no root alias
either: it composes `flagquantum.circuit.Circuit` and reads the unitary back with
the package's own `get_unitary`. `logical_resources.py` imports both: the estimator
and the patch-size helper from the subpackage, and `flagquantum` itself for the
`fq.Circuit` programs it costs. `arithmetic.py` imports both as well: the
constructor and the register map from the subpackage, and `flagquantum` itself for
the `fq.Circuit` register the adder is run on. `examples/README.md` records that
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
needs a commutator norm the caller has to supply, and the block encoding is
built from a dense exact eigendecomposition whose cost is diagonalising the
matrix with nothing here bounding the error of an approximate one, and the logical
resource report counts rather than measures -- no circuit runs, no wall-clock time
or memory is read, and the failure rate is absent because a logical error rate
needs a device's threshold fit, so that number belongs to the device rather than to
the unit. The adder is a construction and its T-cost is the compiler's own Toffoli
rule rather than a second expansion of it written beside the circuit, and it is
Cuccaro's ripple rather than the smaller Gidney-Ekera construction, because
uncomputing by measurement and feedforward needs a mid-circuit measurement path
this repository does not have in a circuit it can cost. The guide holds the full
boundary for each.
