# Program calls and partitioned construction

The source `BloqBuilder.add` accepts a component with its own decomposition,
while `add_and_partition` constructs an AutoPartition and inserts one child
call. The native internal builder now accepts an explicit Program through
`call_program(record, bindings, entry="main")`. `call_partitioned` delegates
to the existing `layout.component_auto_partition` owner and then calls that
Program. Both return the same named-output dictionary as `call_dict`.

Compiler owns import validation, collision handling and live call insertion.
Core retains the sole graph representation. The source decomposition remains
immutable and its selected entry remains nested. Equal definitions are shared
across repeated insertions. Conflicts use deterministic private namespaces,
including calls nested inside imported definitions. Namespaced definitions are
also reused across repeated construction-time inlining. This replaces the
previous inliner-specific namespace selection without changing its gate stream.

`component_imports.prepare_component_import` validates the complete source
Program before definitions or handles are consumed. `call_program` also accepts
a declaration-only typed entry, matching ordinary builder calls. The existing
physical binder still rejects missing bodies, including unused declarations.
`inline_calls` requires an actual typed-call decomposition. Partition shapes,
unused slices, left-only ports and physical layout remain owned by AutoPartition.

## Evidence and boundaries

The source/native test constructs the same AutoPartition calls with permuted
input slices, an unused bystander and left-only outputs. Immediate connection
graphs, full complex tensors and spectator amplitudes match exactly. Existing
expansion yields the same primitive stream. Repeated nested calls preserve one
shared definition and one call per insertion. An occupied namespace, conflicting
child name and declaration-only input are exercised separately.

Two layouts containing a measured controlled child use the previously accepted
source operation directly. Every primitive parameter, operand and feedback key
matches the unchanged source walker. Both measurement outcomes preserve complete
complex amplitudes and return the work bit to zero. The resulting operation is
also appended to DynamicCircuit. The larger existing AutoPartition regression
provides its source model category and call-count witnesses. Pure CNOT layouts
emit one CX with zero workspace, independently of partition bookkeeping.

This closes the nested-import prerequisite for the remaining Q-0156 builder
API. It does not establish completion of the entire source file. Symbolic typed
Programs, broad descriptor admission and the remaining external conversion
facades stay subject to the full-file audit.

The concentrated regression passes 188 cases with four previously accepted
large controlled-swap scenarios deselected. It includes the existing partition
measurement/model tests, complete construction notebook, mutable signatures,
inlining, call builder, graph copying and rebinding. The import module's 22
statements have 100% measured coverage. Strict typing covers four changed Python
files, with Black, Ruff, Compiler scope and architecture checks passing. The
full dependency scan retains 706 existing findings, with zero changed paths.

The frozen `qualtran/_infra/composite_bloq_test.py` also runs unchanged against
the pinned v0.7.0 source: 39 passed and one optional dependency case was
deselected. The separate unchanged upstream notebook test passes as well. Its
nine pytest collection warnings are the source's expected dataclass collection
warnings, not test failures. These source tests validate their own pinned
implementation; they do not replace the native parity and replacement tests
above or close the remaining APIs in Q-0156.
