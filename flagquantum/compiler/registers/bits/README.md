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

`controlled_constant_width.py` reuses controlled XorK emission and the shared
single-zero-control lifetime wrapper. Both directions and polarities keep the
caller control/target live without workspace. Run
`python -m pytest tests/unit/test_controlled_constant_resource_width.py tests/unit/test_controlled_bitwise.py -q`.
[The interface and evidence](../../../../docs/development/CONTROLLED_CONSTANT_RESOURCE_WIDTH.md)
record source self-adjoint child identities and symbolic fallback limits.

`controlled_xor_width.py` binds the existing controlled register-XOR emitter's
atomic Toffolis to through-wire connections. Zero control reuses the shared
control-flip wrapper with both words live. See
[the interface and evidence](../../../../docs/development/CONTROLLED_XOR_RESOURCE_WIDTH.md).

`hamming_weight_width.py` binds count-tree logical junk/output creation and
adjoint consumption. The gate emitter and width graph share the tree and six-step
compression sequence. See [the interface and evidence](../../../../docs/development/HAMMING_WEIGHT_RESOURCE_WIDTH.md).
Run `python -m pytest tests/unit/test_hamming_weight_resource_width.py tests/unit/test_hamming_weight.py -q`.
