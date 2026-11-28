# Component control

This package owns control predicates, multi-control, indexed and controlled component composition.

`controlled_word_x_program` retains an MSB-first opaque control word and scalar
target through the existing predicate owner. Its array counterpart retains the
original outer MultiControlX wrapper. Packing changes no emitted primitives;
both paths share workspace and measured erasure. Run
`tests/unit/test_multi_control_component.py` for all small activation patterns,
literal source hierarchy and primitive/feedback equivalence.

`hubbard_select_component.py` owns literal controlled/uncontrolled Hubbard SELECT
calls and explicit inverse dependencies. The zero-port scalar delegates to
the existing private-carrier owner; no Core effect or backend selection is
introduced. Algorithms and source constructors share this same program. Run
`tests/unit/test_hubbard_select_component.py` for source call comparison,
complex branch amplitudes, clean-work restoration and normal append execution.

`thc_selection_component.py` composes the original fifteen THC SELECT calls from
explicit directional rotation declarations, word swaps and phase wrappers.
It owns one shared angle-word allocation/free. The original optional control
passes through; model-only rotations refuse physical binding. The Formats
adapter supplies constructor data, and shared layout/primitives own the ports
and gates. Run `tests/unit/test_thc_selection_source_constructors.py`.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [controlled_component.py](controlled_component.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_controlled_component.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`multi_controlled_component_program` finds the scalar input/output control by
name, as one THRU declaration or a same-named LEFT/RIGHT pair. This includes
one-qubit opaque ports exposed by `AutoPartition`. The returned
signature puts its QBit array first, with new control before old control; other
ports retain their declared order. Child work and predicate work keep separate
lifetimes, and the predicate owner measurement-erases its workspace. Packing
adds no primitive gates. Exact adjoints of packed callbacks use the existing
adjoint graph owner with explicit inverse Partition pairs and exact child
leaves; the selected child and the callback's packed entry remain distinct.

Run `tests/unit/test_packed_specialized_control.py` for wrapped controls, actual
specialized-adjoint callbacks, all small measurement histories, and the existing
`circuit.append(op)` execution path. Primitive counts and source resource models
are checked separately.

`test_packed_adjoint_control.py` follows the whole existing-control returned
adder through `AdjointWithSpecializedCtrl`, for all four old/new polarities.
The packed current entry and selected helper use the same explicit entry name.
Supply exact inverse pairs for its And compute/erase functions, Partition
functions and child leaves to the existing adjoint owner. The original callback
returns the existing control before the data. With the three-qubit packed data
port tested here, a nonleading control misassigns that returned bit to the data
port while finalizing the source adjoint graph. Native rejects the same
connection. `test_equal_width_control_callbacks.py` shows that equal-width
nonleading ports silently pass type checks but give a callback equivalent to
the intended inverse after an input SWAP. Its returned root, gate counts and
resource model do not expose the difference. Native reproduces that graph;
these cases are defect evidence, not acceptance of the intended operation.

`test_directional_control_callbacks.py` verifies valid `left_only` composition
with a same-named control input/output pair. Existing-control predicates expose
one THRU control array, while packed data is consumed into separate child right
ports. The exact adjoint reverses this flow using supplied inverse Partition
leaves. Caller data wires are not freed; predicate and child work are erased and
released by their existing owners. A renamed control lacking a same-named output
and nesting a directional wrapper as an AutoPartition child remain invalid in
the source. Those failures are diagnosed separately.

`control_component_via_predicate` accepts an explicit positive-control adder at
`controlled[entry]`. Store its called definitions in the same ProgramRecord;
the predicate owner invokes that callback once between compute and erase.
This is the original `ControlledViaAnd` callback path, including directional
input/output ports. Without an entry adder, the ordinary THRU-only hierarchical
control rule still applies. No numerical execution belongs here.

`test_left_only_control_selection.py` checks that direct selection's returned
unwrapped source root differs from its actual wrapped callback entry.
`test_left_only_control_fallback.py` checks two/three-bit default selection,
scalar predicate wrapping, primitive operands/feedback, complete complex
branches, exact adjoint history pairs, clean workspace and append execution.
Compare call counts against the actual source decomposition; its declared
resource model may count the unwrapped root instead. These have equal costs in
this slice because Partition adds no gates. The source fallback zips all data
register declarations with callback outputs. Renamed directional outputs can
therefore fail finalization; native rejects incompatible output-name order
before invoking the callback. This failure is evidence of an upstream boundary,
not successful execution or a corrected reference.

`bookkeeping_component_control_system` owns the control-pass-through protocol
for stored lifetime/rebinding bodies. It follows the pinned
`_BookkeepingBloq.get_ctrl_system`: invoke the same entry once and return the
original control tuple. Allocation, borrowing, free, split, join and partition
are unconditional; do not replace them with predicate-controlled allocation.
The factory verifies the existing ProgramRecord and admits only reachable
allocate/free/split/join operations and calls. It rejects computational or
untyped descendants, including measurement; it adds no IR type or gate.
Unrelated valid definitions may remain shared in the module.

A caller supplies these definitions to its existing ComponentProgramBuilder.
Scalar hierarchical controls can wrap the structured callback using the same
ControlAdder interface as other specialized leaves. Free consumes the binding;
using it again raises the existing builder ownership error. Allocated outputs
require explicit binder ownership transfer. Unmanaged caller inputs cannot be
freed merely because a control is present. The original allocation pool retains
ownership even when Allocate and Free have different dirty flags. Before clean
Free the register must be zero; borrowed-state restoration is an obligation on
the complete computation. Free inserts no numerical reset or state check.

Run `tests/unit/test_bookkeeping_component_control.py` and
`tests/unit/test_controlled_workspace_lifetimes.py` for actual upstream callbacks,
all small measured branches, exact adjoints, complex joint states, independent
source costs and `circuit.append(op)` execution. An output deliberately retained
for its caller can be entangled; only the internally erased work must be zero.

`single_zero_control_width.py` owns X/positive-child/X width connections for
single-control constant arithmetic and XOR. Aggregate word edges preserve
symbolic widths and caller bystanders. The existing scheduler and recursive
query own visitation and caching. Run
`python -m pytest tests/unit/test_controlled_constant_resource_width.py -q`.
[The interface and evidence](../../../../docs/development/CONTROLLED_CONSTANT_RESOURCE_WIDTH.md)
record the concrete and symbolic acceptance boundaries.

`default_component_control_system` dispatches the default request to the supplied
existing total, predicate or classical system factory. Only the exact positive
scalar QBit descriptor selects total control. One-bit words and shaped controls
select the predicate factory. Classical storage identity survives zero width.
Mixed storage and empty specifications fail before invoking any factory. The
selected ProgramRecord and attachment callback pass through unchanged. This
owner performs routing, while each factory retains typed admission, child
semantics and workspace responsibilities. Run
`python -m pytest tests/unit/test_default_component_control.py -q` for source
routing and real attachment-to-execution consumers. General classical stored
child graphs and symbolic execution remain separate acceptance gaps.

Unknown activation in `control_predicate_program` and
`control_component_via_predicate` uses `patterns=None`. Two controls retain an
explicit conjunction declaration. Supply its existing resource identity or
replace it with a concrete body before execution. For three or more concrete
control bits, the pinned source converts its unknown-length activation marker
to all-positive controls. Extension composition preserves that whole-spec rule.
`tests/unit/test_symbolic_predicate_program.py` compares the source connections
and exercises replacement through the same physical consumers.

The bookkeeping factory also accepts existing typed declarations, including
symbolic widths and supported classical or fixed-point descriptors. The original
base control protocol does not inspect the dtype or control specification.
`call_with_unchanged_controls` retains any Sequence container by identity,
including lists and empty tuples. Each callback makes exactly one ordinary child
call with the original inputs. A declaration records that logical connection;
physical binding requires its implementation. Revalidate the factory after
replacing declarations, so computational descendants cannot masquerade as
bookkeeping. This declaration boundary does not certify child semantics.

Run `tests/unit/test_bookkeeping_control_file.py` for the complete base protocol
and all ten original concrete descendants. The complete source-file disposition,
reused execution evidence and independent storage boundaries are recorded in
[the bookkeeping control audit](../../../../docs/development/BOOKKEEPING_CONTROL_FILE_AUDIT.md).

`unary_component_calls.py` appends bounded iteration to an existing
`ComponentProgramBuilder`. Pass one-bit selector handles and a leaf callback
that returns its successor predicate. The leaf owns its other ports and local
allocation/free calls. And predicate ports retain their real directional
lifetimes, so the existing graph adjoint and binder can recycle child and parent
work through one pool. Supply exact And and child inverse definitions to
`adjoint_component_hierarchy`. The primitive-IR and typed-call consumers share
the bounded traversal in `lookup/unary_iteration.py`.

Run `tests/unit/test_unary_component_calls.py` for a complete native construction,
generic inverse, append and all-branch example. It also checks exact source graph
streams, typed-provider replacement and execution with SDK imports blocked.

Generic positive controls on word exchange and selected-word phase preserve the
original call boundaries. The caller owns MSB-first THRU words and the outer
control. Word Split/Join, allocation and release pass that control unchanged.
Controlled bit exchanges reuse the existing packed-AND single-control owner;
controlled MultiControlZ reuses the existing combined-clause predicate owner.
Private conjunction work is restored and measurement-erased by those owners.
The generic path is distinct from a source operation's specialized control
recipe; source constructors may choose different bodies for those paths.

`positive_controlled_multi_x_program` and its Z counterpart preserve a separate
outer scalar control and the original inner control array. They share one
merged-predicate construction rather than flattening those source ports.
The common Program-control callback imports an explicitly selected owner's
complete helper closure; it does not infer semantics from port signatures.
Existing controlled Z and doubling Programs are definition-identical after
that construction reuse.
