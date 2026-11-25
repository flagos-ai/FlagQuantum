# Nozzle inverse with quantum singular value transformation

Run the [notebook](walkthrough.ipynb) or its local CPU workflow from the repository
root.

```sh
python -m examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.main
```

The workflow uses both original matrices and right-hand sides, retains the
24-to-20 Pauli trim for the eight-row example, and uses the original banded
encoding for the sixteen-row example. The inverse polynomial has degree 101
or 255, respectively. Its cutoff uses the smallest singular value expressed in
the block encoding's normalized coordinates.

The example fits a fresh odd symmetric phase sequence for each polynomial.
The bounded Newton solve uses the analytic Jacobian and checks the response on
4,097 points. The phase fitting follows the symmetric-QSP formulation of Dong,
Lin, Ni and Wang. [Paper](https://arxiv.org/abs/2307.12468)

The QSVT circuit synthesizes the original right-hand side, then applies the two
projector-phase sequences needed to recover the real response. The CPU example
packs each alternating block/adjoint sequence as an exact matrix inside a
controlled simulator instruction. Reported resource counts refer to these
represented instructions. The independent sequential reference checks the
complete state, and the SVD reference checks the zero-block inverse action
before comparison with the dense solution.
