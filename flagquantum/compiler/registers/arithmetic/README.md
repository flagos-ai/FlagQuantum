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

`addition_component_program` also accepts explicit `left_dtype` and
`right_dtype` integer descriptors. Their concrete widths must match the width
arguments. Omitting them retains the original unsigned graph. Signed and
Montgomery metadata survive split/join and physical binding, including explicit
Montgomery moduli. The original Add circuit zero-extends a narrower signed input.
The `adjoint=True` option reverses the same logical component sequence before
measured lowering and preserves the original forward graph by default.

`conditional_addition_component.py` composes the complete conditional
sum/difference body from that Add graph, the shared bitwise complement,
hierarchical control propagation and X. It retains ctrl/a/b through words and
the original polarity-switch adjoint. Add owns its measured clean carries.
The existing contributor example now uses this graph and the shared binder.

```bash
python -m examples.algorithms.register_controlled_add_or_subtract
python -m pytest tests/unit/test_typed_addition_component.py tests/unit/test_conditional_addition_component.py -q
```

The example still uses `circuit.append(ConditionalArithmetic())`, verifies
complex amplitudes, checks clean work and appends the opposite-polarity inverse.
Source call/view occurrences, stored-call counts, primitive inventories and QEC
estimates have separate checks. See
[the complete source audit](../../../../docs/development/CONTROLLED_ADD_OR_SUBTRACT_FILE_AUDIT.md).

`addition_width.py` binds existing Add resource identities to connection widths
using `addition_components`. Each logical AND creates a carry, and its adjoint
consumes it. Both words remain caller-owned. The native query supports source
symbolic fallback and Always wrappers. Run
`python -m pytest tests/unit/test_addition_resource_width.py -q` for complete
source cache/order, bystander peaks and DOT width-table acceptance.
[ADDITION_RESOURCE_WIDTH.md](../../../../docs/development/ADDITION_RESOURCE_WIDTH.md)
records lifetime ownership, dtype normalization and separate gate/model counts.

`controlled_addition_component.py` retains the complete CAdd load/Cast/Add/erase
body and every Split/Join call. Each loaded conjunction and Add carry has its
existing measured cleanup owner. Both control polarities and exact stored-call
adjoints reuse the shared And and typed Add dependencies. The existing specialized
control owner also executes the source one-bit outer-control callback in all
four activation combinations. The user example
keeps `circuit.append(ControlledAddition(cv))` and runs coherent control sectors.

```bash
python -m examples.algorithms.register_controlled_addition
python -m pytest tests/unit/test_controlled_addition_component.py -q
```

See [the complete controlled addition audit](../../../../docs/development/CONTROLLED_ADDITION_FILE_AUDIT.md)
for the interface, all-small branch evidence, large factory accounting and
remaining symbolic/outer-control admission boundaries.

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

`constant_addition_component.py` retains the actual AddK allocation, XorK,
Always(Add), XorK and release calls. Its `pattern` is an already encoded unsigned
bit pattern of the supplied integer dtype. Caller-owned numerical encoding
handles signed values. Exact adjoints use Add-adjoint with the same load. Optional
zero/one control affects loading only, so Always(Add) remains unconditional.
Loaded constants and Add carries return clean. The constant append example uses
this graph through the existing register binder.

`out_of_place_addition_component.py` owns an unsigned RIGHT sum and its LEFT
measured consumption. It shares the existing ordered structure and And owner.
Forward binding transfers output ownership with `allow_workspace_outputs=True`.
A separately supplied correct sum needs explicit `input_ownership` for erasure.
The inverse returns only a/b. One-bit integer views stay explicit in native IR,
while the source Cirq importer omits those zero-cost Split/Join calls.

```bash
python -m examples.algorithms.register_constant_addition
python -m examples.algorithms.register_out_of_place_addition
python -m pytest tests/unit/test_constant_addition_component.py tests/unit/test_out_of_place_addition_component.py -q
```

[The complete addition audit](../../../../docs/development/ADDITION_FILE_AUDIT.md)
separates stored calls, measured primitive inventories, source analytical costs
and remaining foreign/symbolic admission boundaries.
