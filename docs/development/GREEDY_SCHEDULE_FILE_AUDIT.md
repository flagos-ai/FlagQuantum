# Greedy instance traversal whole-file audit

The frozen source is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The complete
`_infra/binst_graph_iterators.py` contains `_priority`, the allocation-priority
constant and one generator delegating to NetworkX stable lexicographical
sorting. Its test file contains MultiAlloc and two original tests.

Compiler `_component_trace/width_schedule.py` owns the native queue and width
connections. `schedule_width_steps(..., group_components=False)` uses one global
heap, insertion indices for ties and the same allocation/free priorities.
Default weak-component grouping remains the resource-width consumer's mode.
`registers/component_schedule.py` derives connection-created insertion order
from actual immutable Program arguments, operands, feedback and returns, then
applies the same global owner to each component call before expansion.

The priority branches are zero for boundary nodes, +10**16 for allocation,
-10**16 for free, zero for symbolic widths and output-minus-input width for
ordinary nodes. New direct checks compare every branch to the source, including
positive output-only state, negative input-only effect, neutral CNOT and
symbolic output. Earlier global-queue fixtures cover full graph node order,
independent components, serial calls, partial frees and joined frees. Existing
Program fixtures cover actual source-manager allocation, borrow and release
events, including clean and dirty lifetimes.

## Complete original tests

Both unmodified original test functions run with only their imported sorter
rebound to a test-only source-graph-to-native-width translator. MultiAlloc's
exact eight-node expected sequence passes. Symbolic IntState and repeated Swap
retain a valid complete order. Source graph and SDK objects stay in the test
boundary. Production consumes native connections and Program values.

The translator preserves actual graph node insertion order and each parallel
connection's width. It calls the existing global queue and maps indices back to
source nodes for the original assertions. It neither recalculates priorities
in a second implementation nor modifies original tests.

## Remaining boundaries and resources

Q-0150's complete original test file is accepted. Q-0149 remains open for full
symbolic typed Program graph admission. The standalone numeric/symbolic width
queue and finite executable Program scheduling already have accepted evidence.
Source symbolic dtype and shape metadata must be admitted by their Core owner
before equivalent generic typed graph execution can be asserted.

Source NetworkX traversal is a live mutable-graph generator. Removing a pending
node after the first yield produces its original graph-changed RuntimeError.
Native Program traversal consumes an immutable finite snapshot and returns an
owned order. That representation adaptation is explicit. Source isolated-node
frozenset insertion is process-dependent, so native Program order uses declared
isolated-operation order rather than a cross-process source tie guarantee.

Sorting emits no quantum primitives, allocates no physical work and has no
measurement branches. The executed component/workspace consumers retain their
own instruction streams, complete complex amplitudes, all-small-branch phase
checks, clean-work checks and independent QEC/peak models. Scheduling order and
modeled peak live width remain distinct quantities.

The concentrated batch passed 121 cases with no warnings or skips. Strict
typing of the changed Python file, formatting, lint, Compiler scope, architecture,
multilevel IR and protected API checks passed. The frozen source and original
test blobs are asserted. Logs use
`ftqc-reference-env/g2-greedy-schedule-file-*.log`. The Always and AutoPartition
audit disposition rows are synchronized with their already published concrete
source-capture evidence. Their remaining gaps retain their own parent items.

## Complete implementation closure after symbolic metadata re-audit

Contract prerequisite 1.23 and Compiler commit b63c020d9 close Q-0149's remaining
implementation boundary. The original tests had covered symbolic bit widths but
missed symbolic dtype metadata with a concrete bit width. The recorded probe
uses QFxp(8, f) with symbolic fractional width. The source priority is zero while
the former native priority was eight for an expanding call. The private WidthStep
now retains the signature's symbolic classification before widths are reduced to
edges. Allocation/free priorities still take precedence. Both concrete and
symbolic full typed Program orders match their original source graphs, including
the change in tie ordering. Malformed flags fail explicitly.

The source and native builders both reject symbolic element counts. The frozen
source raises DecomposeTypeError before constructing such a graph. A symbolic
shape is therefore not an executable source graph omitted from this closure.
Existing logical width, fixed-point, bounded and shaped finite-port witnesses,
copy/expansion, original priority tests and physical workspace consumers are
reused. The two source functions and their direct scheduling dependencies are
covered. Source mutation and isolated-node ordering adaptations described above
remain explicit boundaries, not unresolved implementation tasks.

The final concentrated regression passes 1,102 cases without skips and the
interop integration passes 96 cases. Three changed Python files pass strict
typing, formatting, lint, scope, architecture and internal IR checks. Dependency
policy records 689 global findings and zero batch findings. The recent default
suite retains its unrelated example-registry failure after 434 passes and
51 skips. Evidence uses g2-symbolic-priority-. The sort itself emits zero gates,
uses zero work and has no measurements. Existing executed consumer witnesses
retain their full complex states, branch checks and separate resource models.
