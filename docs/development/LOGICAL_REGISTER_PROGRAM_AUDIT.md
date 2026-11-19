# Logical register Program admission

## Ownership and contract

Proposal 103 prerequisite 1.20 and internal IR contract 0.9 separate Core logical
type validation from Compiler physical admission. Core retains immutable type
metadata and existing loose register compatibility. Compiler retains named call
construction, scheduling, copying, expansion, split/join and work lifetimes.
The circuit operation entry remains `circuit.append(op)`.

Logical graphs admit canonical quantum bits, opaque/unsigned/signed words,
bounded unsigned words and fixed-point metadata, with concrete element shapes
and concrete or symbolic bit widths. Fixed-point bit order remains MSB-first.
Split/join requires a concrete bit count. Symbolic-width allocate/free preserves
one owned register value and needs no physical wires during construction.
Physical lowering and binding validate every declared type, including empty
arrays and uncalled definitions. Existing numerical descriptor limits remain
explicit at these boundaries.

## Frozen source comparison

The reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Source test blob
`83dfa234bc95c8310f1d3d8526fbb2aa9b538a7f` and composite implementation blob
`b0317f675c96f9cce1581fcbb7030c48ab32a529` are checked before the original
`test_type_error` executes unchanged.

`test_logical_register_program.py` constructs the actual source
TestMultiTypedRegister and TestQFxp fixtures. Native named calls, producer ports,
indices, scalar/array shapes and graph copies match their source counterparts.
The signedness failure follows the two preceding successful fixed-point and
unsigned calls. Both mismatches retain the source's two consumed input
connections and no produced outputs. The fixtures are declarations for this
path. Their numerical decomposition is outside this test-file requirement.

For five symbolic-width families, scalar, two-element and empty arrays compare
immediate source/native call traces. Native copying and inlining retain the
logical type and element ownership. The frozen source empty-output copy defect
is tested separately and preserved as a recorded upstream limitation. Native
empty mappings copy successfully. Physical rejection includes empty symbolic
arrays and an uncalled symbolic function, where flattened-element checks alone
previously missed unsupported metadata.

## Numerical and resource evidence

This change admits logical metadata and emits no new primitive gates. It does
not assign physical qubits or a finite numeric cost to symbolic words. Numerical
execution evidence comes from the existing concentrated component suite,
including full complex CNOT operators, register permutations, controlled
partitions and all measured Program/Cirq branches with work restored. Primitive
counts and source resource estimates retain their prior separate reporting.

## Remaining boundaries

Symbolic element shapes, broader classical/field descriptors, arbitrary SDK
wrappers and source summary formatting remain separate ledger gaps. Fixed-point
and bounded logical admission does not extend the physical numerical profile.
The source's empty-output copy failure remains recorded. No frozen source files
are modified.

## Local validation

The combined 25-file regression passes 914 cases without skips, including the
full native construction notebook, complex operator/measurement comparisons,
logical descriptor contracts, workspace and resource tests. Seven changed
Compiler/test files pass strict typing, Black and Ruff. Compiler scope and
architecture pass. Internal IR validation passes all 15 rules and 43 diagnostic
witnesses. Dependency policy retains 706 existing findings and zero findings
on changed paths. Logs are retained as `g2-logical-register-*.log` in the local
reference evidence directory.
