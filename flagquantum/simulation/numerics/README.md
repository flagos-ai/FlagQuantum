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
explicit NumPy `RandomState`, accepts an integer seed, or uses the global RNG
when passed `None`, and reproduces the source's real orthogonal QR and
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

## Sampled QSP polynomial checks

`qsp_sampling` owns explicit-RNG QSP coefficient draws and sampled GQSP pair
consistency. `qsp_grid` owns FFT-grid maxima, scaling, and assertions through a
caller-owned FFT provider or, when omitted, a lazy optional SciPy FFT import.
Neither a random check nor a finite grid certifies
the entire unit circle; a source counterexample is retained. See
[the executable example, source failures, and resource boundaries](../../../docs/development/QSP_POLYNOMIAL_CHECKS.md).
Run `python -m pytest tests/unit/test_qsp_sampling.py tests/unit/test_qsp_grid.py -q`
with the pinned reference environment. These are numerical inspection helpers,
not a QSP circuit synthesis implementation.

## Matrix assertions

`matrix_assertions` owns phase-sensitive default-norm comparison and the separate
unitary assertion that ignores global phase. The latter reuses `unitary_distance`
and must not replace full-amplitude conformance. See
[the executable example and source assumptions](../../../docs/development/MATRIX_ASSERTIONS.md).
Run `python -m pytest tests/unit/test_matrix_assertions.py -q` in the reference
environment to compare both assertions with unchanged upstream functions.

## QSP complements and phases

`qsp_complement` owns root-based complementary coefficients; `qsp_phases` owns
numerical phase recursion. Neither emits a GeneralizedQSP circuit. The phase
recursion uses the source rotation definition without importing Compiler.
See [the runnable numerical chain and retained source failures](../../../docs/development/QSP_COMPLEMENT_PHASES.md).
Run `python -m pytest tests/unit/test_qsp_complement.py tests/unit/test_qsp_phases.py -q`
with the pinned reference environment for coefficient, matrix and rotation checks.


## Whole-source closure

The polynomial evaluator and both matrix assertions retain upstream NumPy
operation order, dtype/subclass dispatch, failures and source test behavior.
See [module acceptance and resource interpretation](../../../docs/development/POLYNOMIAL_ASSERTION_CLOSURE.md).
With pinned references, run the two focused test files named there. These
helpers do not synthesize circuits or claim that phase-insensitive distance
certifies exact amplitude equivalence.
