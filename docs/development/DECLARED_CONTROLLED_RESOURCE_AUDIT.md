# Controlled declared resource propagation

This audit covers the previously missing generic declaration dependency of
Q-0158 `Controlled.build_call_graph`, source lines 655–672. The frozen source is
Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`, controlled.py blob
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28`. Copyright 2023 Google LLC,
Apache-2.0. The source obtains the child's declaration with the same symbolic
allocator, controls each declared callee, and accumulates equal identities.

## Owner and interface

Compiler `_component_trace/controlled.py` introduces the lazy `ControlledCall`
resource node alongside the existing `AdjointCall`. It stores a native child
identity and encoded activation bits, or a symbolic count for unspecified bits.
Its `callee_counts` method calls the existing shared declaration accumulator.
An optional control factory supplies specialized child identities. Nested
control selection can be supplied through that factory. Default nodes retain
the child identity and activation, and their labels match the original C or
C[n] spelling.

The same SymbolAllocator reaches the forward declaration. Mapping and set
inputs, symbolic multiplicities, shared nodes, zero edges and merged specialized
children use the existing graph machinery. Missing decomposition errors and
factory errors propagate. A caller may use the existing graph keep policy to
retain an unavailable atomic node as a terminal.

This resource identity uses encoded activation rather than foreign dtype/shape
objects. Typed Program records remain authoritative for register metadata,
physical control selection, lifetimes and execution. Native constructors do
not import or retain Qualtran objects. The resource node composes the existing
symbolic_call_graph API and introduces no parallel graph representation.

## Source and consumer evidence

`test_declared_controlled.py` directly executes the unchanged source default
Controlled wrapper for mapping/set declarations, three activation patterns,
symbolic counts and specialized controls. Equal specialized child identities
merge with the original multiplicities. The native allocator identity is
verified through the actual forward callback.

A recursive graph with a shared child compares every directed edge weight and
weighted terminal count with the original source graph. A separate zero-edge
case checks retention, and missing declarations and specialization failures
are verified. Atomic or unavailable children remain visibly unavailable unless
the consumer explicitly keeps them as terminals.

Physical direct export, native append, complete complex amplitudes and observed
primitive streams are independently checked in
`CONTROLLED_REGISTER_EXPORT_AUDIT.md`. Declaration operations emit zero physical
instructions. Their symbolic counts require a caller-provided leaf metric for
an analytical estimate. An unknown leaf is assigned no invented primitive cost.
Physical measurement branches and workspace retain the previously accepted
predicate/binding owners and their concentrated regressions.

## Whole-source disposition

Generic declared-child control propagation is implemented. Full Q-0158/Q-0159
acceptance remains open for the other control admission and symbolic metadata
boundaries enumerated in `CONTROLLED_INTERFACE_SOURCE_AUDIT.md`. The source
helper's enriched exception text is expressed through native exception types
and existing propagation. Exact text equality across framework exception
hierarchies is outside the native interface contract.

## Completed batch checks

All 4047 concentrated cases and 96 SDK integration cases pass. The two new test
files contribute 23 focused cases. Strict typing passes four changed Python
files, and Black, Ruff, Compiler/Ecosystem scope, architecture and IR checks pass.
Dependency analysis records 687 whole-repository findings with zero in the
changed paths. Logs and the observed resource receipt use the
`g2-declared-controls` prefix. SDK import observations include opaque matrix
instructions, which retain their explicit `any` identity and granularity.
