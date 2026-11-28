# Integer arithmetic

`sparse_index_component.py` owns the source sided TopLeft identity and banded
Add/signed-AddK index graphs. Exact inverse pairs retain arithmetic call order
and measured cleanup. It consumes native widths and bounds, not SDK objects.
Run `tests/unit/test_sparse_matrix_component.py` for the oracle replacement,
complete small operators and the separately recorded symbolic-bound storage gap.

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

`subtraction_component.py` stores both complete subtraction bodies.
`subtraction_program(a_dtype, b_dtype=None)` preserves a and writes a-b, retaining
unsigned casts, three OnEach complements, the shared equal-width Add and the
original prefix lifecycle. Signed a copies its sign through shared fanout;
other integer inputs zero-extend. `subtract_from_program(dtype)` preserves a and
writes b-a through typed BitwiseNot/Add/BitwiseNot. Add owns its measured carries.
All prefixes and carries return clean. The source's loose QAny/QUInt view edges
stay separate from strict dtype-check admission. The existing append example uses
these native graphs through the shared binder.

```bash
python -m examples.algorithms.register_subtraction
python -m pytest tests/unit/test_subtraction_component.py tests/unit/test_fanout_component.py -q
```

See [the complete subtraction audit](../../../../docs/development/SUBTRACTION_FILE_AUDIT.md)
for original tests, source declarations and remaining generic admission gaps.

`negation_component.py` composes BitwiseNot/AddK(1) over one typed integer word.
The explicit adjoint reverses the actual child calls and uses the measured
AddK-adjoint, while a second forward negation also restores the logical input.
Single-control polarities and multi-register predicates share the existing
control hierarchy. `controlled_negation_program` accepts the same Register and
ControlRegister descriptors as the predicate owner, including row-major arrays
and signed activation words. Predicate and AddK workspace have separate owners.
All work is measurement-erased and returns clean. The encoded native one-bit
signed permutation remains distinct from the source signed +1 load failure.

```bash
python -m examples.algorithms.register_constant_addition
python -m pytest tests/unit/test_negation_component.py -q
```

The append example checks plain negation, coherent negative control and the
exact adjoint. [The complete negation audit](../../../../docs/development/NEGATION_FILE_AUDIT.md)
records the source defect, separate model/primitive counts and remaining gaps.

`integer_square_component.py` exposes the native sided integer-square Program.
Forward preserves THRU unsigned `a` and creates RIGHT double-width `result`.
Inverse consumes the matching LEFT image and frees its zeroed storage. The
column owner controls product/carry work and shared measured And calls.
Bind forward with output storage followed by scratch in the clean pool, and
`allow_workspace_outputs=True`. For standalone inverse, identify the result
storage using `input_ownership` so it can be freed. Nested callers transfer this
ownership through the existing builder. Run
`python -m pytest tests/unit/test_integer_square_component.py -q` for the source
signature, logical output lifetimes, complex amplitudes and erasure histories.

`integer_product_component.py` and `sum_of_squares_component.py` create and erase
owned unsigned result images through the same column graph as Square. Product
preserves two input words. SumOfSquares preserves its row-major shaped input.
`column_result_component.py` owns only result lifetimes and typed views, with
three current consumers. Product/carry scratch remains in the bit-column owner.
The physical binder accepts each shaped input element as a separate wire tuple.
Forward clean pools place result storage before scratch. Inverse result words
must be identified in `input_ownership`. The existing append entry is exercised
with both flat roots replaced. Run
`python -m pytest tests/unit/test_column_result_components.py -q` for both owned
result workflows. See the multiplication audit for unavailable source root
expansions and separately reported primitive/model counts.

`product_accumulation_component.py` preserves PlusEqualProduct's occupied THRU
accumulator. It sends the live product column word directly to shared Add or its
exact inverse, then erases products/carries. It reuses the typed column steps
with weighted-column XOR consumers. No product output word is allocated. Run
`tests/unit/test_product_accumulation_component.py` for coherent occupied inputs,
wraparound, measurement histories, work cleanup and ordinary append replacement.

`subtraction_declaration.py` owns generic Subtract/SubtractFrom dtype defaults,
unsigned projections, signatures and ordered direct models. The subtraction
catalogue consumes these declarations. Concrete execution delegates to the
existing `subtraction_component.py`, with caller-owned a/b and clean prefix/carry
lifetimes. `subtraction_control.py` derives both scalar control polarities
through the same graph, reusing CAdd and controlled BitwiseNot. Signed prefixes
retain the original controlled fanout with atomic Toffoli leaves. Model Toffoli
cost and emitted atomic gates are reported separately. Run
`python -m pytest tests/unit/test_subtraction_control.py tests/unit/test_subtraction_foreign_file.py -q`.
The ordinary `circuit.append(op)` entry remains in `register_subtraction.py`.

`bounded_increment_component.py` binds the existing unit-step emitter beneath
the original atomic ModAddK boundary. Controls are ordered QBit elements and the
unsigned word is MSB first. Padding states are fixed. Private clean scratch and
feedback remain owned by the existing emitter. Run `test_bounded_increment.py`
and `test_hubbard_stored_graph.py`. Original ModAddK has no primitive recipe, so
keep its five-Add model separate from literal native gates.

`nested_controlled_constant_component` composes two-control XOR loading around
the existing unconditional Always(Add) or its exact inverse. Both scalar
controls and the data word are THRU; the constant word and carry work are
private. Constants are already encoded by the caller. The normal append path
uses the same bound primitive IR. Run
`tests/unit/test_nested_controlled_constant_component.py` for complete source
primitive/feedback comparisons, all small raw histories, complex states, zero
work and constructor replacement. Source modeled Toffoli costs stay separate
from emitted atomic CCX and Clifford/T counts.

`specialized_controlled_addition_component` retains the original three-call
conjunction/CAdd/erasure specialization for ordered arrays of at least two
controls. It preserves both control polarities and the exact positive CAdd
inverse. `additional_inverse_control_component` owns the different source path
for Controlled(Adjoint(Add/CAdd)): generic call propagation, atomic Toffoli and
actual source-refused controlled And declarations. It must not substitute the
forward specialization for that generic inverse path. One-wire Add operands
remain scalar as in the original Cirq importer. Run
`tests/unit/test_specialized_controlled_addition_source.py` for complete source
layers, primitive feedback, complex operators, all small measurement histories,
entangled spectators, zero work and ordinary append execution.
