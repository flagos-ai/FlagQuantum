# Gate and measurement primitives

This package owns boolean/classical/quantum gate dispatch, temporary AND, controlled swaps, measurement cleanup.

`controlled_z_word_component.py` owns the shaped-control/opaque-word signature
and exact Split/MultiControlZ/Join wrapper that phases the first system bit.
Formats and Algorithms share this signature; the existing multi-control owner
supplies clean predicate work and measured erasure. Run
`tests/unit/test_thc_selection_source_constructors.py` for primitive source
comparison, all small complex branches, spectator preservation and normal append.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [and_compute.py](and_compute.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_addition.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`controlled_rotation.z_power_with_control_ir` follows original ZPow control
selection through measured CZPow, with zero-control X wrapping and an exact
exponent adjoint. Its clean work and reusable result slot are explicit. The
[measured GQSP guide](../../../../docs/development/GQSP_MEASURED_CONTROL.md)
contains the dynamic append example and distinguishes original Bloq leaf parity,
direct Cirq semantics and modeled costs.

`swap_component.py` owns ordinary/controlled word-exchange declarations and the
stored plain Split/atomic-SWAP/Join body. Controlled execution delegates to the
existing exact Fredkin component owner. Logical port order, MSB-first bit pairs
and caller storage survive binding without workspace or feedback. Symbolic
signatures and direct models are available independently of physical dimensions.
Run `python -m pytest tests/unit/test_swap_foreign_file.py -q` for original
factories, foreign library resolution and the ordinary append workflow.

`always.py` owns emitter control bypass and directional signatures;
`always_component.py` stores one typed child call. Generic direct resource
forwarding belongs to `compiler/_component_trace/always.py`, where native child
identities and the caller's symbol allocator remain independent of emission.
The complete source dispositions and primitive/model separation are recorded in
[the Always audit](../../../../docs/development/ALWAYS_FILE_AUDIT.md).
