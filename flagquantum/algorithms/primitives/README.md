# Algorithms primitives

Reusable quantum primitives that the algorithm modules share. This layer owns
the quantum Fourier transform, phase estimation, state preparation, and oracle
synthesis, and all four of them ship today.

A primitive is admitted here when more than one algorithm module needs it, or
when it is a public unit callers use directly: state preparation shipped on the
second ground, with no consumer inside `flagquantum/` at all, and the Fourier
transform shipped with one, phase estimation. This package is not a
general-purpose quantum toolkit, and a construction only one workflow uses stays
in that workflow's module until a second one needs it.

## Where to start

- `qft.py`: the quantum Fourier transform and its inverse, emitted as a circuit
  fragment that callers append to a circuit they already hold, and the primitive
  phase estimation consumes.
- `phase_estimation.py`: phase estimation over a controlled unitary, with the
  resolution and the success bound the counting register buys.
- `state_preparation.py`: a uniform superposition, and an arbitrary state built
  from a classical amplitude vector by uniformly controlled rotations.
- `oracle.py`: the reversible classical building blocks the oracle units are
  composed from -- a multi-controlled X and a bit-string comparator -- and the
  truth-table synthesis of a phase or bit oracle on top of them.
- `types.py`: the callable protocols the primitives are written against.
- `__init__.py`: the public primitives surface.

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
