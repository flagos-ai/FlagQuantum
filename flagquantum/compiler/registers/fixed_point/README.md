# Fixed-point arithmetic

This package owns polynomial approximations, reciprocal and inverse-square-root steps.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [fixed_point_polynomial.py](fixed_point_polynomial.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_fixed_point_polynomial.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`fixed_point_square_component.py` composes the retained D24/D28 terms through
shared weighted columns. Its signature preserves both pure-fraction THRU inputs
a and b, creates a RIGHT result, and consumes the matching LEFT image on erase.
The existing paper construction reads a and preserves b. The typed result owner
keeps the fixed-point descriptor while measured And calls restore scratch.
Run `tests/unit/test_fixed_point_square_component.py` for the stored/expanded
call, complex-state, measurement-history and ordinary append workflow.
