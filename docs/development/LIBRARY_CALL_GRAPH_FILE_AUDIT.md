# Complete library call-graph source closure

The frozen source is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Complete
`dev_tools/qualtran_dev_tools/all_call_graph.py` and its test file correspond to
Q-0055 and Q-0056. Tests verify both original Git blobs.

## Ownership and execution

Compiler owns `symbolic_call_graph_union` and the shared adjacency traversal in
`_component_trace/call_graph.py`. Single-root graph queries use the same traversal
and retain their existing root normalization, cutoff, counts and allocator
behavior. The library query retains concrete roots and generalizes callees,
expands shared nodes once across all examples, and creates a fresh symbol
allocator per declaration. These are the actual frozen library-call-path rules.
Although the original helper accepts an allocator, its internal callee query
does not forward it. This observed behavior is preserved for the library query.

Integration's `tools/development/library_call_graph.py` consumes existing lazy
`ComponentExample` records, logs each name, evaluates factories sequentially and
delegates graph traversal to Compiler. The generic entry accepts domain declarations and policies. The
`resource_library_call_graph` entry supplies the complete source terminal and
five-step policy using native resource keys. That policy maps to `ignore_register_partition`,
`generalize_control_values`, `generalize_rotation`, `ignore_allocation` and
`ignore_known_cliffords`. The domain supplies its direct model declarations because native components
have no common foreign base class. Explicit T, Toffoli and rotations remain
terminal. Missing capabilities produce the source warning and remain leaves. Tests exercise that native profile
and typed resource identities independently of any foreign object.

The return value is weighted native adjacency with isolated nodes retained.
NetworkX is only used at the reference-test boundary to compare the unchanged
original acceptance test. Framework execution does not depend on NetworkX or
Qualtran. Declared graph multiplicities are model counts; they are not primitive
instruction measurements. The existing `circuit.append(op)` entry remains served
by its existing component binding and execution owners.

## Complete reference evidence

All 393 original library examples produce exactly matching sets of 3,693 nodes
and 8,550 weighted edges through the native traversal, with zero cycles. The
original library test runs unchanged against the native graph consumer after
adapting the discovery root and adjacency representation. Its original
terminal predicate and generalizers are captured intact from the actual source
entry, not reimplemented for comparison.

Some factories create fresh callable identities or identity-equality objects.
The test constructs each input once and replays the identical roots into both
implementations. Independently reconstructing such inputs compares different
identities and cannot prove graph equality. The source and native declaration
callbacks, dependencies, weights and complete graph remain separately evaluated.

Native tests also cover repeated roots, shared-node first visits, fresh symbols,
empty libraries, isolated roots, root/callee normalization differences, terminal
cutoffs, capability refusal, signed raw declaration weights and factory failures.
The raw graph layer preserves symbolic and signed source multiplicities; physical
resource validation remains with its existing consumers. Existing single-root
models and source-reference graph tests are included in the batch regression.

## Boundaries

Cycles raise the existing native graph ValueError. The original builder retains
a cyclic adjacency and its later acyclicity test rejects it. This is an explicit
earlier native diagnostic, not a claim of identical cyclic-input failure timing.

Original warnings for unavailable leaf models remain visible in the complete
source corpus. Successful graph assembly does not prove a physical decomposition
for those leaves. The existing native capability-refusal contract is preserved.
Terminal predicates are responsible for domain-specific missing-model warnings.

The complete graph comparison exercises source model declarations through the
native generic traversal. It does not claim that every source node has a native
physical decomposition. Those module-level acceptance gaps remain in the main
ledger. No amplitude, measurement or workspace claim is inferred from model
adjacency. Existing component execution and resource evidence is reused.

The final concentrated run passed 374 cases. It retains 45 original
missing-model and source/dependency deprecation warnings. Strict mypy passed
all three changed Python files. Black, Ruff, Compiler and Integration scope,
architecture, internal IR, public API and whitespace checks passed. The native
README model example executed successfully. Existing source tests, symbolic
counts, resource classification, Drawer and component-example execution passed.
