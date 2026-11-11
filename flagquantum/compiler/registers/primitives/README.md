# Gate and measurement primitives

This package owns boolean/classical/quantum gate dispatch, temporary AND, controlled swaps, measurement cleanup.

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
