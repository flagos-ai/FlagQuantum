# Native flexible QPE

Run the complete original-sized experiment from the repository root:

```bash
python -m examples.synthesis.tutorials.advanced_tutorials.high_level_modeling_flexible_qpe.overview.main
```

`walkthrough.ipynb` covers every source code stage. `ansatz.py` contains the
native quantum functions, original operator, fresh eigendecomposition and
per-power repetition bindings. `reference.py` constructs independent direct
Pauli-product matrices, exact exponentials, inverse Fourier matrices and the
controlled-SWAP Circuit. `workflow.py` checks full physical states, numeric
marginals and finite-shot readout. `plots.py` renders actual observed phases.

The exact source eigenvector index is retained. Observable index zero is the
least significant qubit; the dense NumPy matrix follows the same convention.
The duplicate ZZ terms retain source order for the finite product formula.
No precision increase, Hamiltonian replacement or postselection is introduced.

A phase QNum is all fractional. Scalar gates may access its raw encoded bits
without changing the numeric format; this does not authorize unsigned arithmetic
borrowing or sliced fixed-point computation. The same data register participates
in approximate and exact QPE before the phase comparison. Dropped data and phase
registers remain a possibly entangled environment, never clean workspace.
The swap-test statistic is therefore labeled an overlap diagnostic rather than
asserted to be fidelity between independently prepared pure phase states.

The original sampling calls leave shot count implicit. This local example
explicitly uses 1024 draws with seeds 4100 and 4101. Historical source counts are
not acceptance targets. There is no hosted job, added SciPy dependency, backend
selection, hardware, optimality, speedup or scalability claim.
