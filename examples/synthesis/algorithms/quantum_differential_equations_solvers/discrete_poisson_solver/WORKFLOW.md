# Discrete Poisson solver

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.main
python -m examples.synthesis.algorithms.quantum_differential_equations_solvers.discrete_poisson_solver.comparisons.circuit
```

The native program prepares the retained 64 spatial amplitudes, applies bounded
Type-II sine/cosine kernels to two three-bit views, estimates the diagonal
operator with six phase bits, applies the retained 64-entry Y-rotation table,
and uncomputes phase estimation and spatial transforms. All 16 ordered Pauli
terms, including both identity coefficients, remain unchanged. Since these
terms commute, powers of the one-step evolution equal the corresponding longer
exact diagonal evolution.

The phase register uses unsigned six-fractional-bit storage for the native QPE
contract. Its raw bits and rotation-table addressing equal the original integer
register; interpreting the output as a numeric integer requires multiplying by
64. The indicator and spatial register are named outputs. The phase register is
retained, not falsely declared clean: finite-resolution inversion can entangle
it with the indicator. Compiler workspace is zero.

The independent Circuit uses a Householder input preparation, Fourier-derived
sine/cosine matrices, the finite-difference eigenvalue formula, controlled
diagonal evolution and an analytic inverse DFT. Observable indices count from
the least significant register bit; the reference explicitly preserves the
source's axis and bit assignment rather than correcting it to a preferred PDE
convention. Validation checks every complex amplitude, including absolute phase,
for the default and two signed nonzero inputs against both native and original
programs. Separate coherent-control, identity-phase, inverse, bit-order, alias,
invalid-input and resource-budget tests protect the composition.

Native default resources are 13 physical bits, zero workspace, 7,728 operations
and depth 4,770. Original resources are 13 bits, zero workspace, 5,632 operations
and depth 2,657. Default independent-reference errors are below 6e-14. The native
resource overhead is retained as an optimization gap, not a performance claim.
This accepts the quantum program; the surrounding Notebook workflow remains
pending.
