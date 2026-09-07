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
