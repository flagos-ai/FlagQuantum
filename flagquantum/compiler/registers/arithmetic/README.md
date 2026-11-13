# Integer arithmetic

This package owns addition, subtraction, integer products, squares and their component costs.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [addition.py](addition.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_addition.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

[Factorized addition programs](../../../../docs/development/ADDITION_COMPONENTS.md)
compose typed CNOT/AND calls over the same logical sequence as the gate emitter.
Physical binding retains the existing append workflow; explicit numerical
suppliers can contract large closed graphs without a full addition matrix.
Run `tests/unit/test_addition_component.py` for the complete contributor scenario.

`addition_width.py` binds existing Add resource identities to connection widths
using `addition_components`. Each logical AND creates a carry, and its adjoint
consumes it. Both words remain caller-owned. The native query supports source
symbolic fallback and Always wrappers. Run
`python -m pytest tests/unit/test_addition_resource_width.py -q` for complete
source cache/order, bystander peaks and DOT width-table acceptance.
[ADDITION_RESOURCE_WIDTH.md](../../../../docs/development/ADDITION_RESOURCE_WIDTH.md)
records lifetime ownership, dtype normalization and separate gate/model counts.

CAdd connected resource widths live in `controlled_addition_width.py`. The owner
keeps caller words live across controlled load, the shared Add child and erasure.
See [the interface and acceptance](../../../../docs/development/CONTROLLED_ADDITION_RESOURCE_WIDTH.md).

Constant-add connected resource widths live in `constant_addition_width.py`.
The owner composes allocation, XorK, Always(Add), unload and release.
See [the interface and acceptance](../../../../docs/development/CONSTANT_ADDITION_RESOURCE_WIDTH.md).

Controlled constant-add widths extend `constant_addition_width.py`, preserving
control liveness across Always(Add). Controlled XorK belongs to the bits owner,
and both polarities share the single-zero-control width owner.
See [the interface and acceptance](../../../../docs/development/CONTROLLED_CONSTANT_RESOURCE_WIDTH.md).

Out-of-place sum lifetimes live in `out_of_place_addition_width.py`. Physical
emission and width connections share `out_of_place_addition_structure.py`.
AND creates output bits and its exact adjoint consumes them, with one explicit
low-output allocation/free. See
[the interface and acceptance](../../../../docs/development/OUT_OF_PLACE_RESOURCE_WIDTH.md).
