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


`modular_exponentiation_component` composes the original IntState, reverse
exponent-bit traversal and shared controlled multipliers. Its RIGHT result
belongs to the caller; private multiplication work must close. The source
constructor uses executable bodies, while the existing component tutorial
explicitly retains declaration-only leaves for large logical drawings. The
numerical make_on path delegates to the shared Program and an explicit classical
scratch pass while retaining its original workspace and slot budgets. See
`docs/development/RSA_MODEXP_FILE_AUDIT.md` for source inverse and large-width
boundaries. Run `tests/unit/test_legacy_modular_exponentiation_factories.py`
and `tests/unit/test_register_component_infrastructure.py` for this change.
Exact inverse flags reverse shared logical calls in the comparator, controlled
modular add, constant, scaled and multiply owners before physical lowering.
Scoped ModExp constructor recipes reuse these inverse bodies and consume the
logical result only on the prepared image. The original signed comparator
inverse port-name failure remains an explicit negative boundary.

Scaled modular addition imports each distinct reduced constant's Program once
per construction and calls that local definition for every original source bit.
The compute/add/erase order, repeated source operations, negative scales, exact
inverse pairs and measured feedback remain unchanged. There is no cross-call
cache or second definition registry. Full Program and bound-record comparison
against published 4551a7e11 verifies this construction reuse before reusing its
source amplitude/resource evidence. Source models and literal primitives retain
their separate owners; this change does not remove gates or lower cost estimates.

`modular_addition_program` and `modular_doubling_program` accept `adjoint=True`
to reverse their existing child calls using exact addition, comparison and
constant-load inverses. Forward Programs are unchanged. Doubling returns a
different physical layout, so inverse consumers must use actual output bindings
and transfer pool ownership when a source release touches caller storage. Run
`tests/unit/test_ecc_arithmetic_adjoints.py` for original streams, all small
complex columns, erasure histories and private cleanup.
