# Matrix inversion with fresh QSVT phases

Run the [notebook](walkthrough.ipynb) or the local workflow from the repository
root.

```sh
python -m examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.overview.main
```

The workflow regenerates the original seeded 4 by 4 matrix. Its Pauli LCU
contains all sixteen terms and uses four selector bits, with the original
right-hand side `[0, 1, 2, 3]`. It checks the actual block encoding and
calculates both the bare and normalized singular-value condition numbers.

The explicit route keeps degree 151 from the original formula and fits fresh
symmetric QSVT phases to a shifted Chebyshev inverse polynomial. The
precision-driven route chooses its degree from the 0.01 error bound on the
unscaled inverse operator. Both execute the real response from the two exact
phase streams using a local Circuit operation and recover the signed solution
from the zero-block state. Independent sequential and SVD references check the
complete state and the projected inverse action. The CPU circuit represents
the phase streams as matrix operations.
