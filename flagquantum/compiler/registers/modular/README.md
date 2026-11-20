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

Concrete negation and subtraction graphs live in `modular_negation_component`,
`controlled_modular_negation_component`, `modular_subtraction_component` and
`controlled_modular_subtraction_component`. They reuse the shared modular
addition, exact constant adjoint, zero-control, fanout and measured conjunction
owners. `ModularSubtractionExample` binds forward graphs through
`circuit.append(op)` and retains the independently verified explicit inverse
emitters for `adjoint=True`. Read
`docs/development/MODULAR_SUBTRACTION_FILE_AUDIT.md` for residue-domain promises,
source failures, complete definition ownership and separate primitive/model
resource evidence.

The complete modular multiplication graph family uses separate doubling,
controlled multiplier, single-window, window traversal and output-allocation
owners in the corresponding `*_component` modules. Shared clean-bank lifetime
and measured erase composition belongs to `lookup.clean_lookup_component`.
The three existing register multiplication/doubling examples keep their append
entry and logical aliases. Read
`docs/development/MODULAR_MULTIPLICATION_FILE_AUDIT.md` for all 45 source
definitions, exact arithmetic dependencies, source failures and the model-only
lookup adjoint's separate executable resource accounting.
