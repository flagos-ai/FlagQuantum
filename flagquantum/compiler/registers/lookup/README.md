# Table lookup and selection

This package owns qROM, QROAM, selected swaps, unary traversal and measured unlookup.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [qrom_table.py](qrom_table.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_qrom_table.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`clean_lookup_component` binds a scalar power-of-two dataset to an executable
stored component. Forward target and junk ports transfer owned clean banks to
the caller. The measured erase consumes those matching banks and returns its
scratch clean. It reuses `source_qroam_clean` and `qroam_clean_erasure`, whose
primitive counts remain separate from the original model-only adjoint cost.
Run `python -m pytest tests/unit/test_clean_lookup_component.py -q` for source
forward primitives and every raw small lookup/erase measurement history.
