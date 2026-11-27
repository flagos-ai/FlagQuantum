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

`finite_differences.py` owns the single shared central-difference quotient and
the displacement derived from delivered precision. Public gradient requests and
conformance oracles reuse these functions; neither Core contracts nor this module
select a different method or claim the quotient is exact. Run
`tests/unit/test_finite_differences.py` and `tests/test_gradient_api.py` for the
reported method, step, scalar objective and parameter-shape behavior.

`probabilities.py` owns shared dense probability extraction, ordered joint
reduction, normalization and parity reconstruction. It consumes existing
tensors and retains the delivered arithmetic, tolerance and allocation rules.
Runtime selects backend/catalog paths and admits requested marginal sizes.
Run `tests/unit/test_measurements_marginal_reduction.py` and
`tests/test_modern_measurements.py` for the local readout workflow.

## Pauli readout draws

`probabilities.py` owns Pauli sampling normalization and draw/bit reconstruction.
The two numerical calls preserve normalization before Runtime creates its
generator. Request resolution, generator ownership and result metadata stay
in Runtime.
