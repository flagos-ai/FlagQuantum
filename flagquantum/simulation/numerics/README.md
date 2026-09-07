# Simulation Numerics

This package owns reusable numerical primitives used by simulation engines.
It currently contains the experimental Double-Single FP32 representation,
arithmetic, and conformance runner.

It does not select devices, authorize precision modes, schedule execution, or
claim hardware support. Core owns precision requirements and plans; Runtime
owns selection and evidence. Start in `double_single.py` for arithmetic changes
and run `tests/unit/test_double_single.py` plus
`tests/test_double_single_conformance.py`.
