# Modular arithmetic

This package owns controlled modular arithmetic, multiplication, windowed methods and inverse composition.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [modular_addition.py](modular_addition.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_modular_addition.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

Concrete source-shaped stored graphs live in `modular_addition_component`,
`controlled_modular_addition_component`, `modular_constant_component` and
`scaled_modular_addition_component`. They compose the shared typed arithmetic,
comparison, integer basis and conjunction owners. The existing
`ModularAdditionExample` catalog binds these graphs through `circuit.append(op)`
for residue-range moduli. The original composite full-ring refusal remains because the frozen
source loader cannot encode that modulus. ModAddK retains its independently
implemented full-ring permutation. Read `docs/development/MODULAR_ADDITION_FILE_AUDIT.md` for interfaces,
ownership, source failures and separate primitive/model evidence.
