# Bit operations

This package owns xOR, bitwise transforms, population count, signed encodings and register rotations.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [bitwise.py](bitwise.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_controlled_bitwise.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`bitwise_width.py` reuses the XOR/constant-XOR emitters and the existing OnEach
boundary to bind arithmetic resource identities to automatic connection widths.
All words remain caller-owned and no workspace is allocated. Run
`python -m pytest tests/unit/test_bitwise_resource_width.py -q` for frozen-source
cache/order, symbolic fallback, signed range and source-width DOT comparisons.
[BITWISE_RESOURCE_WIDTH.md](../../../../docs/development/BITWISE_RESOURCE_WIDTH.md)
records native identity normalization and execution/resource evidence.
