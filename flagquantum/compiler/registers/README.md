# Register decompositions

This internal Compiler package owns register-level decompositions into native
FlagQuantum IR, compile-time wire binding and component cost declarations.
User workflows continue through `circuit.append(op)`.

## Package ownership

| Package | Responsibility | Entry module |
| --- | --- | --- |
| [primitives/](primitives/README.md) | Boolean/classical/quantum gate dispatch, temporary AND, controlled swaps, measurement cleanup | [and_compute.py](primitives/and_compute.py) |
| [bits/](bits/README.md) | XOR, bitwise transforms, population count, signed encodings and register rotations | [bitwise.py](bits/bitwise.py) |
| [layout/](layout/README.md) | Partition, cast and bit views, clean and borrowed workspace ownership | [partition.py](layout/partition.py) |
| [control/](control/README.md) | Control predicates, multi-control, indexed and controlled component composition | [controlled_component.py](control/controlled_component.py) |
| [arithmetic/](arithmetic/README.md) | Addition, subtraction, integer products, squares and their component costs | [addition.py](arithmetic/addition.py) |
| [modular/](modular/README.md) | Controlled modular arithmetic, multiplication, windowed methods and inverse composition | [modular_addition.py](modular/modular_addition.py) |
| [modular/inversion/](modular/inversion/README.md) | Arithmetic, ordering, parity, exchange and termination steps of the inverse algorithm | [inverse_algorithm.py](modular/inversion/inverse_algorithm.py) |
| [finite_field/](finite_field/README.md) | Polynomial-basis field operations, linear maps, product and inverse chains and erasure | [coefficients.py](finite_field/coefficients.py) |
| [comparison/](comparison/README.md) | Equality and ordering with coherent, measured and mixed operands | [less_than.py](comparison/less_than.py) |
| [lookup/](lookup/README.md) | QROM, QROAM, selected swaps, unary traversal and measured unlookup | [qrom_table.py](lookup/qrom_table.py) |
| [phase/](phase/README.md) | Phase gradients, programmable rotations, phase costs and measured QFT | [phase_gradient.py](phase/phase_gradient.py) |
| [fixed_point/](fixed_point/README.md) | Polynomial approximations, reciprocal and inverse-square-root steps | [fixed_point_polynomial.py](fixed_point/fixed_point_polynomial.py) |
| [sorting/](sorting/README.md) | Compare/exchange networks, permutations, duplicate detection and set combinations | [bitonic_sort.py](sorting/bitonic_sort.py) |
| [preparation/](preparation/README.md) | Uniform and rank-aware preparation and reflections | [uniform_preparation.py](preparation/uniform_preparation.py) |
| [resources/](resources/README.md) | Cross-component symbolic resource formulas consumed by domain cost modules | [symbolic_resource_model.py](resources/symbolic_resource_model.py) |

## Dependency boundaries

- Core owns register types, ordered signatures, symbolic parameter semantics and IR.
- Register components own decomposition, register bit order and workspace cleanup.
- `_component_trace/` owns call attribution, resource traversal and query policy.
- `algorithms/` composes components into algorithm-level operations.
- `lowering/`, `artifacts/` and `formats/` own target transformations and output.
- Runtime organizes execution. Simulation supplies numerical kernels.

Each subpackage initializer contains documentation only. Import a component from
its owning module. Source attribution stays in the implementation headers.

`component_bindings.py` groups Core indexed bindings into scalar/object-array
collections and visits them in declared port order. A caller supplies existing
value identities and owns connection/consumption effects. The shared Core dtype
policy runs after each callback, retaining the pinned source's failure effects.
The module supplies binding collections and traversal for the finite Program
component builder below. Direct numerical graph traversal remains pending.

## Contributor path

Run an existing lookup consumer and the resource comparison example from the
repository root. These use concrete and symbolic tables and separate model
estimates from emitted instructions.

```bash
python -m examples.algorithms.register_qrom_table
python -m examples.algorithms.resource_model_comparison
python -m pytest tests/unit/test_qrom_table.py tests/unit/test_register_signature.py -q
```

Choose the owning package above, change the component, then run its focused tests
and the applicable source-parity regression. Preserve logical ports, measurement
branches, complex amplitudes and workspace ownership. Public APIs and serialized
IR contracts follow their existing change-proposal process.

## Relocation map

The [complete module map](../../../docs/development/REGISTER_PACKAGE_LAYOUT.md)
records all 191 moves. The internal flat module paths have been replaced at all
checked-in executable consumers. Test filenames retain their existing scenario
and upstream-reference identities.

Internal emitters use functional names. The callable map in the module guide
records 40 renamed functions and their owners. Source schedules, register
contracts and cost formulas retain their existing behavior.
General typed component graph execution remains separate integration work.


## Finite gate-call expansion

`program_calls.inline_program_calls` expands verified single-block calls in
the existing Core ProgramRecord. Each invocation has its own SSA mapping. An
explicit stack handles deep finite call chains, and parameter/measurement
feedback bindings follow that same mapping. DNF conditions are canonicalized
after substitution, including classical aliases. Core independently validates
the result and attests any retained public-circuit payload.

The focused path is `tests/unit/test_program_call_lowering.py`. Its existing
Add/QROM append consumer is reused with hierarchical emitter replacement,
checking unchanged source gate dependencies, all four measurement histories,
complex amplitudes, workspace cleanup and separate primitive/model resources.
Workspace remains explicit in function inputs and outputs.

## Typed register expansion

`register_programs.lower_register_program` validates the complete Core module,
flattens declared words in port/row-major/MSB order and reuses the finite call
inliner above. Owned zero-width words have no wires after verification.
Split/join preserve ordered tuples and emit no gates. A reordered return remains
explicit, and Core payload restoration refuses a hidden output permutation.
Gate-only `inline_program_calls` refuses unlowered register functions.

Run `pytest tests/unit/test_register_program_lowering.py` for the shortest
contributor path. The existing Add/QROM append consumer runs unchanged with
three-level packed hierarchy replacement, primitive equality, all measurement
histories and independent resource checks. Finite mutable construction and
rebindings are implemented below. Physical clean/dirty wire binding is
implemented in the balanced profile below. Direct numerical graph traversal
retains separate acceptance work.

## Finite register construction

`component_builder.ComponentProgramBuilder` receives an ordered Signature and
real immutable typed function definitions. `inputs`, `call_dict`, `call_tuple`
and `call` preserve named/ordered port collection conventions. The shared
binding collection helper allocates one fresh SSA value per element. The shared
binding visitor preserves callback-before-dtype checks and consumption prefixes.
Calls and returns are recorded directly as existing Program records. Core
verifies every function body at finalization, including retained failed calls.
A failed zero-port call can retain a valid empty instance, matching the source.

Run `pytest tests/unit/test_component_program_builder.py` for identity, arrays,
zero-width ownership, failed calls/returns and the unchanged real Add/QROM append
consumer. These retain exact primitive instructions, all measurement histories,
workspace cleanup and separately reported models. Only the concrete quantum
bit/opaque/unsigned profile is admitted. Balanced clean/dirty physical workspace
is supported below. Additional descriptors and numerical traversal remain
separate.


## Builder word and bit rebindings

`component_rebindings.py` supplies real typed helper functions consumed by the
same builder's `split` and `join` methods. Each logical bit retains an owned word
handle. Helpers convert those words through the existing gate SSA rebindings.
The extra logical split/join operations lower to zero physical primitives and
zero workspace. These helpers carry executable bodies, not signature stubs.

Run `pytest tests/unit/test_component_rebindings.py`. Fifteen original ownership
and failed-join scenarios accept the native observation adapter. Four remaining
source scenarios compare connection order and source model cost against native
zero-primitive lowering. Complex spectator/permutation checks include a real X
call. Existing Add/QROM consumers retain all measurement histories and resources
after explicit split/join around every returned word, including width zero.

## Explicit workspace construction

`layout/component_workspace.py` supplies real typed allocation and release
function bodies. `ComponentProgramBuilder.allocate` and `free` use ordinary
finite calls and existing per-element SSA ownership. Explicit dtype overrides
`n`, zero-width scalar words retain ownership, and stale free calls reserve an
instance before failing. Both clean and dirty flags survive in the leaf record.
The admitted profile requires concrete quantum bit/opaque/uint descriptors and
built-in boolean flags. Source metadata construction has a wider domain.

Run `pytest tests/unit/test_component_workspace.py tests/unit/test_program_workspace.py`.
Sixteen unchanged source builder scenarios accept the native observation adapter.
The tests also verify real helper bodies, dtype changes after split/join, call
forwarding, source resource models and independent Core rejection.

`lower_register_program` retains its gate-only contract and rejects lifetimes.
The physical entry below consumes lifetime graphs using the existing
`layout/workspace.py` pool as its physical lease authority. Graph acceptance
establishes typed consumption. Clean zero-state and dirty original-joint-state restoration
need separate numerical evidence. Source tensor/classical allocation uses zero
even for dirty allocation, while its Cirq path borrows unknown state. Keep those
entry points distinct when extending execution.

The existing physical pool also supports `free_wires` for ordered partial
release and cross-allocation joins. It validates all requested wires before
mutation, retains each wire's allocation pool, and preserves release order for
subsequent allocation. Whole-lease `free` releases the still-live remainder.
Zero-width leases retain identity and use whole-lease release. The method accepts
physical wire ownership only. Program SSA remains responsible for stale logical
handles, and numerical restoration remains an emitter obligation.

Run `pytest tests/unit/test_workspace_partial_release.py tests/unit/test_register_workspace.py`.
The physical entry below consumes these pool operations.
`lower_register_program` remains the separate gate-only entry.

For the physical scheduling prerequisite, the shared
`_component_trace.width_schedule.schedule_width_steps` accepts
`group_components=False`. It uses one global priority queue in source graph
node insertion order. Existing peak estimates retain default weak-component
grouping. The source graph adapter must supply actual graph insertion order,
including isolated nodes, instead of assuming builder call positions. Run
`pytest tests/unit/test_component_physical_schedule.py` for original-source
physical visitation comparisons. Program physical binding is implemented below.


## Physical register binding

`register_binding.bind_register_program` accepts actual verified Program
functions, input-element wire tuples and explicit clean/borrowed pools.
`component_schedule.local_operation_order` derives source connection-created
node order for each definition, schedules locally through the existing greedy
kernel and then reuses finite call expansion. The existing workspace pool binds
allocation, partial release and cross-allocation joins. Primitive operands,
parameters, fresh measurement slots and boolean feedback become existing IR.
The result also retains ordered output words and immutable lifetime events.

Run `pytest tests/unit/test_register_physical_binding.py`. The unchanged Add/QROM
append consumer uses real private-work lifetimes and explicitly maps local
classical slots to its original slot bindings. It checks source primitive
counts, per-wire dependencies, every small measurement history, complex
amplitudes and clean-zero work. Borrowed work is verified with entangled external
spectators. Pool events and complete local operation order match six original
source graphs. Repeated calls keep local lifetime order before expansion.

The default profile has quantum boundary ports and balanced nonempty work.
Explicit `allow_workspace_outputs=True` transfers returned allocated wires and
their subsequent lifetime to the caller. The pool retains their ownership during
binding, including partial and nested output lifetimes. Explicit
`input_ownership=((wire, dirty), ...)` adopts pre-owned input wires into
their clean or borrowed origin pools. Releasing and reallocating these wires
requires output transfer; returning an unreleased input does not. Measurement-produced CBit outputs use parallel classical-slot tuples. External
classical initialization, destructive workspace retirement and isolated source
frozenset tie parity remain open.
Binding ignores the optional restoration payload and executes the actual graph.
See
[the interface and evidence](../../../docs/development/PHYSICAL_REGISTER_BINDING.md)
for bit order, output semantics, slot alpha-renaming and separate resource counts.


Run `pytest tests/unit/test_register_output_transfer_contract.py tests/unit/test_register_output_transfer.py`
for original-source output order, allocation-without-free events, full complex
spectator states, all measured Add branches, nested live leases and partial
release/reuse. Outputs contain wire tuples in declared port order. The caller
owns cleanup and eventual borrowed-state restoration after transfer.

Run `pytest tests/unit/test_register_input_pool.py
tests/unit/test_workspace_adoption.py` for input adoption, source manager
reuse, measured erasure branches and pool-origin validation.


Run `pytest tests/unit/test_register_owned_measurement.py` for destructive QBit
measurement, nested owned CBit identities, reordered arrays, existing bool slot
isolation and all complex source-instrument branches. Pool-owned destructive
measurement defaults to rejection. The explicit measured_workspace policy can
retire clean wires permanently or reset them before pool reuse. Borrowed wires
require original-state restoration and refuse both policies. See
`tests/unit/test_register_workspace_retirement.py` for the shortest execution path.

`component_classical.parity_word_function` builds the source CBit-array THRU and
parity RIGHT signature over the existing owned Program operations. Binding uses
the shared classical XOR assignment, retaining predecessor slots for owned
successors. `read_classical_register` forwards ownership and exposes a shareable
bool snapshot of the immutable slot without emitting an instruction. Run
`pytest tests/unit/test_register_classical_parity.py` for nested components,
zero-input parity, both feedback polarities, complex measurement branches and
actual trajectory/batched execution through unchanged `circuit.append(op)`.

`discard_word_function(name)` wraps `discard_classical_register` with one CBit
LEFT port. Core enforces one consumption. Binding deletes the owned handle,
emits no instruction, and retains earlier bool snapshots of the immutable slot.
This is classical Discard from pinned Qualtran `basic_gates/discard.py`.
External CBit initialization follows proposal 103 prerequisite 1.17. Physical
workspace retirement follows prerequisite 1.16. The shortest test path is
`tests/unit/test_register_classical_discard.py`, including measurement, parity,
discard and subsequent conditional feedback through the existing append path.

`component_measurement.measure_x_word_function` composes H and destructive
MeasureZ using existing register operations. The physical measured wire retains
its outcome. `component_feedback.classical_cz_word_function` forwards the owned
CBit control and applies existing conditional CZ through a bool snapshot.
`finite_field.product_erasure_program` composes these with shared Parity, Discard
and typed Split/Join. Resolved binary-field descriptors retain nominal polynomial
identity and MSB-first physical words. The source polynomial loop reverses bits
locally. Run `tests/unit/test_field_erasure_program.py` for the source-builder
replacement and complete branch/spectator scenarios. The existing flat erasure
emitter retains caller-owned physical clearing and reusable parity-slot behavior.


`entry_bindings.entry_input_layout` owns finite physical entry widths and
concrete external CBit validation. `bind_register_program(classical_inputs=...)`
initializes each CBit with one fresh constant classical XOR slot. The tuple
aligns with every flattened LEFT element, using None at quantum positions.
Quantum zero-width words retain their distinct dtype and own no classical slot.
The shortest test path is `tests/unit/test_register_classical_inputs.py`, with
source row-major assignment, nested identity/Parity/Discard, measurement slot
isolation and actual `circuit.append(op)` feedback.

`component_graph.iter_component_calls` visits the existing typed entry operations
in pinned graph order and exposes named predecessor/output binding collections.
`copy_component_program` replays those calls with fresh SSA bindings through the
same builder. Callee definitions remain shared immutable records. Both validate
the entire module first. Logical traversal counts CBit edges at their source
one-bit width, while physical scheduling retains zero quantum width for CBit.
Run `tests/unit/test_component_graph.py` for row-major arrays, zero-width words,
directional lifetimes, original erasure graphs and copied graphs. The Add and
field-erasure suites exercise copied entries through the unchanged append path
with complete small measurement histories and separate primitive/model counts.
Selective single-level and bounded recursive expansion are implemented below.

`component_expansion.expand_component_calls_once` expands selected calls by one
level of an existing typed call body. Primitive bodies remain leaves. The
predicate receives an occurrence label and the existing Operation. Returned
labels align with the new entry operations, preserving retained occurrences and
assigning inserted labels above the old maximum. These labels record compiler
provenance, while Core Program SSA remains authoritative for ownership and
connections. `expand_component_calls` repeats passes until no decomposition is
accepted, with the source depth bound including the final no-change probe.

Logical traversal inspects only the current component layer when classifying
Allocate/Free. A composite allocation/free wrapper keeps ordinary component
priority until expanded. Physical scheduling retains its existing helper
classification. Run `tests/unit/test_component_expansion.py` for original-source
predicates, retained/gapped labels, empty identities, arrays, output permutations,
atomic fallback, failure propagation and depth limits. The same suite exercises
actual directional field-erasure graphs, full small measurement histories and
the existing append path. The Add suite also runs recursive expansion before
its unchanged primitive, branch, workspace and separate model checks.

`component_adjoint.adjoint_component_program` reverses a typed call entry using
an explicit mapping to existing specialized adjoint definitions. It flips entry
ports and reconnects SSA in reversed logical visitation order. Exact per-side
signatures are checked before construction. The caller supplies the semantic
adjoint bodies, including measured arithmetic cleanup. Run
`tests/unit/test_component_adjoint.py` for original-source Split/Join and
Allocate/Free graphs, ordered H/T conjugation, all small Add inverse branches,
complex spectator amplitudes and clean workspace restoration.

`component_adjoint.adjoint_component_hierarchy` derives shared adjoint definitions
from stored call-only bodies while preserving explicit specialized child mappings.
Original child definitions remain available to forward callers. Generated names
avoid all existing definitions. Primitive/mixed bodies need an explicit target.
`_component_trace.program_call_graph.program_component_call_graph` counts actual
stored calls with the existing resource graph, keep policy and depth semantics.
Run `tests/unit/test_component_hierarchy_adjoint.py` and
`tests/unit/test_program_component_call_graph.py` for hierarchical source graphs,
shared leaves, direct/weighted calls, exact measured Add cleanup and model closure.

`component_control.control_component_program` belongs to Compiler. It takes an
existing typed-call Program and a child-name to controlled-adder mapping. A
scalar positive QBit is prepended to the all-THRU entry, with the source's
collision-free control name. Adders call existing specialized definitions and
can rename or pack ports, matching the original get_ctrl_system protocol.
They return data in the original child's output order. Control and data successors
are threaded through the existing SSA builder in logical visitation order.
Specialized bodies own gate semantics, private work and measurement cleanup.
The wrapper owns no numerical execution and adds no gates or work. Other control
profiles require separate acceptance.

`component_control.control_component_hierarchy` derives one shared positive-control
definition per unmapped stored call-only child. Explicit child adders take
precedence and own their specialized semantics. Generated definitions preserve
forward definitions, use collision-free internal names and thread control
through the same one-level implementation. Primitive or mixed children require
an explicit adder. The derived entry cannot be overridden by an adder mapping.
Core Program remains the graph authority and final module verification applies
to retained callers as well as generated definitions. Builder-created split/join
and lifetime helpers inside explicit adders are retained with each derived body.
Run `tests/unit/test_component_hierarchy_control.py` for original-source hierarchy
connections, shared leaves and diamonds, separate modeled/primitive counts,
exhaustive small measured histories and sampled larger graphs.

`component_control.control_component_via_predicate` follows ControlledViaAnd
around that shared positive hierarchy. Callers supply existing Register ports
and ControlRegister activation words. Control ports precede the data ports and
must have distinct names. Single-bit control admits scalar or shape-(1,) QBit,
with X before and after a negative activation. The multi-bit path calls the
directional predicate, the positive body and the measured predicate erase once
each. Specialized adders preserve the predicate and own their separate work.

`control.component_predicate.control_predicate_program` owns those directional
ports. Controls retain row-major element order and MSB-first word order. Forward
allocates n-1 clean qubits and returns the first n-2 as opaque `junk` plus a QBit
`target`. Both are live logical outputs. Erasure consumes their correct computed
values, applies the shared And emitter in reverse stage order, resets and frees
the outputs, and returns the original controls. Split/join remains gate-free.
Core currently admits unparameterized bit, opaque and unsigned quantum words
on this path. Signed, ones-complement, bounded, Montgomery, symbolic and
classical descriptors need separate Core admission before typed execution.
Explicit outer control names are supplied by the caller on this internal API.

Run `tests/unit/test_component_structured_control.py` for pinned signatures,
named graph connections, primitive operands/feedback, independent resource-model
closure, complete small measurement histories, complex amplitudes, clean work,
exact adjoint round trips and the unchanged `circuit.append(op)` consumer.

`control.component_extension.extend_component_control_system` owns the stored
ControlledViaAnd.get_ctrl_system extension protocol. It takes the original
forward Program, specialized positive adders, existing named controls and new
control descriptors with their activation words. New clauses precede existing
clauses in one combined predicate. Combined names use the source collision rule
against forward data ports. New descriptor names label their tuple positions.
The returned module and adder share its definitions through the normal builder.
The adder pops old controls from the supplied data dictionary, invokes the
combined wrapper once, and returns new controls separately from old controls
and data successors. Both sets retain shape and MSB-first ordering. The wrapper
owns predicate lifetimes and measured erasure. Original forward semantics and
the exact specialized adjoint bodies retain their existing owners.

`control.specialized_component_control` owns packed controls around a stored
positive single-control implementation. Its input Program declares a leading
scalar THRU QBit control. The helper replaces that port by a QBit array, computes
the original And/MultiAnd predicate, calls the positive implementation once and
measurement-erases the predicate. Prefix junk has the source's QBit-array shape.
Array/word conversions reuse the directional predicate builder without gates.
The two-control factory accepts the old and added activation values and returns
the existing ControlAdder protocol. It packs new then old controls, consumes the
old named control from its mutable input dictionary and unpacks the successors.
Predicate outputs stay owned across the child, whose private work is separate.

`control/component_control_selection.py` selects an existing single-control
Program by activation, combines an existing scalar control through the shared
packed helper, or delegates unchanged descriptors to the supplied fallback
owner. The direct adder preserves source input-dictionary mutation and declared
output order. Positive specialization is required. `control/component_adjoint_control.py`
pushes only explicitly selected single-bit polarities through exact adjoints.
It reverses the actual returned-adder graph, preserving data and appended-control
order. The source callback's inline graph is retained as a local stored adapter,
which `expand_component_calls_once` can inline at the caller. Generic inverse
control remains owned by the supplied inverse fallback. These modules own
Compiler selection and graph transformations, with no execution or new Core types.
Run `tests/unit/test_component_control_selection.py` and
`tests/unit/test_component_adjoint_control.py` for the source dispatch, callback,
complex-state, work-zero and existing append path examples.

`layout/component_auto_partition.py` adds a packed typed entry around an existing
child entry using the existing AutoPartition layout. External words are concrete
scalar opaque registers. Child array elements use row-major, MSB-first order.
Partition and inverse-partition calls reuse builder split/join ownership and add
no primitive gates or work. Unused segments remain live through the child.
left_only consumes packed inputs and exposes child right ports, with physical
input ownership transfer required for frees. The original child entry remains
available for specialized-control unwrapping. Select the packed callback entry
explicitly when matching the source's wrapped adder and unwrapped return value.
See tests/unit/test_component_auto_partition.py and
tests/unit/test_specialized_control_boundaries.py for complete examples and the
original shaped-control and directional failure boundaries.

`component_tensor_expansion.py` applies the tensor-preserving/full-expansion
selection rule through the existing component expansion owner. Tensor providers
supply an explicit callee classifier; full mode never queries it. Simulation
continues to own binding/contraction. See
[component tensor expansion](../../../docs/development/COMPONENT_TENSOR_EXPANSION.md)
for composite-entry scope, source correspondence, failures and measured append
verification. Its `expand_component_root_for_tensor` entry explicitly wraps a
typed root in one existing builder call before applying the same policy; partial
mode may retain the root itself. Already-composite callers keep the original
entry. Original signatures and definitions survive, with no numerical execution
or new resource ownership.

`component_zero_initialization.py` wraps an existing typed entry with source-ordered
clean word preparations and one child call, preserving RIGHT outputs and explicit
ownership transfer. It reuses existing allocation definitions and owns no tensor
execution. See [zero-input initialization](../../../docs/development/COMPONENT_ZERO_INITIALIZATION.md)
for namespace/metadata rules, original failure boundaries and local tests.

The [stored-unitary GQSP guide](../../../docs/development/GQSP_STORED_UNITARY.md)
connects these existing hierarchy, control, exact-adjoint and physical-binding
paths to the same dynamic append consumer. It records full source leaf equality,
all small measurement histories and separate emitted/model counts for a
noncommuting shared H/phase component.
