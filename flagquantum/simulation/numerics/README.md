# Simulation Numerics

This package owns reusable numerical primitives used by simulation engines.
It contains backend-portable complex arithmetic and the experimental
Double-Single FP32 representation, arithmetic, and conformance runner.

It does not select devices, authorize precision modes, schedule execution, or
claim hardware support. Core owns precision requirements and plans; Runtime
owns selection and evidence. Start in `complex_arithmetic.py` for portable
complex operations or `double_single.py` for extended-precision arithmetic,
and run `tests/unit/test_double_single.py` plus
`tests/test_double_single_conformance.py`.

## Dense unitary inspection

`unitary_polynomial.evaluate_polynomial_of_unitary_matrix` evaluates ascending
coefficients with an integer power offset. `unitary_distance` owns the explicitly
phase-insensitive comparison metric. These NumPy inspection functions do not
synthesize operations, estimate resources, select execution paths, or provide
PyTorch gradients. Full complex-amplitude checks remain authoritative for exact
and controlled-operation conformance. See
[the runnable example and source boundaries](../../../docs/development/UNITARY_NUMERICS.md).

With the pinned reference environment, check the adapted functions against
unchanged upstream implementations:

```bash
python -m pytest tests/unit/test_unitary_polynomial.py tests/unit/test_unitary_distance.py -q
```

## Random real symmetric inspection matrices

`random_hermitian.random_hermitian_matrix(dimension, random_state)` consumes an
explicit NumPy `RandomState` and reproduces the source's real orthogonal QR and
symmetrization sequence. It does not sample an arbitrary complex ensemble.
See [source conditions, runnable example, and refusal behavior](../../../docs/development/RANDOM_HERMITIAN.md).
Run `python -m pytest tests/unit/test_random_hermitian.py -q` with the reference
environment. This adds no production Cirq dependency or new execution engine.

## Jacobi-Anger approximation

`jacobi_anger_coefficients` owns cosine/sine Laurent coefficients;
`jacobi_anger_degree` owns the source threshold search and its symbolic formula.
Both accept explicit Bessel evaluators instead of importing an undeclared SciPy
production dependency. The symbolic path reuses Core's parameter classifier.
See [the executable composition and exact adaptation boundaries](../../../docs/development/JACOBI_ANGER.md).
Run `python -m pytest tests/unit/test_jacobi_anger_coefficients.py tests/unit/test_jacobi_anger_degree.py -q`
with the pinned reference environment for source comparisons.
