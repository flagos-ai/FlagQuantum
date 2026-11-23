# Subtraction and fanout constructor dependency audit

## Complete source scope

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The complete
`arithmetic/subtraction.py` and `mcmt/multi_target_cnot.py` sources and their
original tests were read, alongside their native declarations and physical
owners. Five frozen source/test/resolver blobs are verified in
`ftqc-reference-env/g2-subtraction-constructor-intake-20261123.json`.
Complete original source-file evidence in SUBTRACTION_FILE_AUDIT.md and the
existing fanout/control suites is reused.

| Constructor or dependency | Authoritative native owner | Binding |
|---|---|---|
| Subtract | `arithmetic/subtraction_declaration.py` and `subtraction_component.py` | Caller-owned a/b, b becomes a-b, signed a sign-extends into a clean prefix |
| SubtractFrom | The same declaration and component owners | Caller-owned equal-width a/b, b becomes b-a |
| MultiTargetCNOT | `bits/fanout_component.py` and `bits/bitwise.py` | Caller-owned scalar control and dirty target word, shared source tree |
| Symbolic Cast dependency | Existing `layout/register_cast.py` signature | Logical source/target views without a physical body |

Compiler `formats/legacy_subtraction_factories.py` delegates constructor defaults,
dtype validation and signatures to the existing subtraction declarations.
`formats/legacy_fanout_factories.py` selects the existing shared fanout tree.
Source constructor identifiers remain interchange metadata. No source SDK object,
new public schema, numerical kernel or arithmetic algorithm is introduced.

Words are MSB first. Reversed physical words preserve caller ordering.
Montgomery descriptors and their modulus metadata retain the existing owners.
Subtraction prefix work and measured Add carries have distinct clean lifetimes.
Both input words remain outputs. Fanout preserves arbitrary dirty target inputs,
uses no workspace and is exactly self-adjoint.

## Direct dependency closure and symbolic declarations

The complete source constructor dependency group reuses the accepted Add,
BitwiseNot, OnEach, XGate, conjunction and bookkeeping catalogues. Narrow signed
Subtract uses the actual shared MultiTargetCNOT prefix extension and reversal.
Named constructors and directly decomposed libraries emit the same comparable
primitive dependencies, operands and measurement feedback.

The original symbolic subtraction CompositeBloq uses Cast metadata before its
symbolic OnEach and Add leaves. The existing Cast signature already supports
symbolic views, while its executable word body requires concrete width. The
bookkeeping adapter now returns that existing logical signature as a declaration
for symbolic dtypes. Physical binding retains the missing-body refusal and all
quantum/classical permission and concrete-width rules. Both scalar and shaped
symbolic Cast libraries are tested. The symbolic CompositeBloq can consequently
resolve all stored calls at serialized depths one and two. Depth zero omits its
connections and retains the missing-connection diagnostic.

Both whole source files' ten original examples are accounted for at retained
serialized depths. Generic control and adjoint wire wrappers remain independent
open scope. Existing native controlled subtraction, negative polarity, exact
adjoints and original foreign recursion failures retain their accepted evidence.
Zero-width fanout and arithmetic retain their original or native physical refusal.

## Complex execution and resource accounting

The physical constructor tests compare original source streams with named and
directly decomposed native libraries, including original measurement slots and
feedback predicates. Independent original tensor contractions check all complex
amplitudes with an entangled spectator. Every small measurement history and
paired inverse history restores the expected unnormalized phase and clean work.
Subtract is its own inverse. SubtractFrom is reversed through the existing Add.
Complex64 and complex128 trajectories execute through circuit.append(op).

Fanout checks complete coherent dirty-target states, exact involution and source
tree operands on reversed targets. Its actual primitive inventory contains
2n-1 CX instructions. The unchanged source QEC model treats the complete fanout
as one atomic Clifford, so the analytical model reports one. Both are asserted
separately. This category granularity difference remains explicit.

Subtract and SubtractFrom emit 4(n-1) T/TDG and n-1 measurements/resets for target
width n. Source QEC T and measurement categories agree. Signed narrow Subtract
also owns target-width minus source-width clean prefix wires. Source Clifford
models retain their existing Add and fanout granularity limitations and are not
substituted for the actual primitive inventories.

## Local acceptance evidence

Concentrated affected regression passes 1,395 cases. Local runtime integration
passes 145 cases. The coverage selection passes 203 cases, measuring 100 percent
statements and branches in both new constructor owners and the affected
bookkeeping adapter. Strict typing covers all five changed Python files.
Black, Ruff, Compiler scope, repository language, architecture, IR and public
API checks pass. The dependency checker still exits one with exactly the same
720 baseline findings.

The unchanged original source suites pass 92 cases with eight original
symbolic/protocol/strict-dtype skips and four source xfails. Five frozen blobs
and the native Kickmix 0.1 extension and callable Simulator are verified.
Evidence is retained under ftqc-reference-env/g2-subtraction-*-20261123,
including the intake JSON, original log, regression log, integration log,
quality JSON, coverage log and runtime JSON.

Three additional complete identifiers bring built-in constructor coverage to
40 of 303, leaving 263, including generic Adjoint/Controlled/Power and partial
OnEach. The broader Q-0876 resolver parent remains open. Pending Proposals
116-119, original source defects and whole-repository/GPU/performance limits
retain their existing dispositions.
