# Algorithms primitives

Reusable quantum primitives that the algorithm modules share. This layer owns
the quantum Fourier transform, phase estimation, state preparation, and oracle
synthesis; the Fourier transform is the primitive that ships today, and the
others arrive with the algorithms that need them.

A primitive is admitted here only when at least two algorithm modules need it.
This package is not a general-purpose quantum toolkit, and a construction only
one workflow uses stays in that workflow's module until a second one needs it.

## Where to start

- `qft.py`: the quantum Fourier transform and its inverse, emitted as a
  circuit fragment that callers append to a circuit they already hold.
- `__init__.py`: the small public primitives surface.

## Boundaries

Primitives build circuits from other circuits and from classical data. They do
not define circuit or operator semantics, compiler passes, numerical kernels,
runtime selection, provider lifecycle, or any performance claim. Numerical
kernels stay in `simulation/`, which owns gate matrices and state evolution;
a primitive calls those through the ordinary circuit API instead of
reimplementing them or carrying a second execution path.

Every primitive in this package is demonstration scale unless its own module
docstring says otherwise. Nothing here carries a capacity, throughput, or
quantum-advantage claim.

For a small change, add a focused public function, check it against an
independent reference, and run:

```bash
python -m pytest tests/unit/test_algorithms_qft.py \
  tests/unit/test_algorithms_package.py -q
python tools/check_architecture.py
```
