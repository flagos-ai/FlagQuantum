# Component control

This package owns control predicates, multi-control, indexed and controlled component composition.

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
