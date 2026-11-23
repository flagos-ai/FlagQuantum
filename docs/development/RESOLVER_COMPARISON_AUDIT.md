# Comparison constructor dependency audit

## Source scope and interfaces

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch covers the complete
`arithmetic/comparison.py` constructor family, its private concrete half helper,
and its directly serialized sign-conversion, multi-control Pauli and Z/CZ leaves.
Eleven frozen source/test/resolver blobs are checked in the batch runtime receipt.
Existing complete numerical, notebook, connection-trace and physical evidence in
COMPARISON_FILE_AUDIT.md, SIGN_EXTENSION_FILE_AUDIT.md and the component suites
remains authoritative. Old references there to pending Proposal 108 are historical.
Proposals 108 through 115 have since been approved and implemented.

Compiler formats owns constructor adaptation. Existing comparison components own
constant, equality, tree, linear, controlled and directional half decompositions.
Bits owns redundant sign prefixes. Control owns literal predicates and measured
conjunction erasure. Numerical simulation stays with its existing owner. The
consumer remains `circuit.append(op)`. No external source objects enter Compiler.

All words use MSB-first order. Through operands remain caller-owned. Mixer and
single-qubit flags produce RIGHT outputs and their inverse consumes the exact
LEFT image. Half comparisons produce an unsigned n+1-bit difference and a flag,
then release them through specialized measured erasure. The source ge/lt inverse
returns exchanged a/b physical aliases. Both aliases and the underlying exact
complex-state restoration are retained. LinearDepthGreaterThan keeps unsigned
port descriptors even when its signed flag is true.

## Constructor closure

| Source group | Native implementation and boundary |
|---|---|
| LessThanConstant, EqualsAConstant | Existing seed/prefix and MCX constant owners. Concrete thresholds retain source validation and upper-bound behavior. Symbolic metadata declares a logical function. |
| GreaterThanConstant | Logical declaration, matching the source's model-only class without an executable decomposition. |
| BiQubitsMixer, SingleQubitCompare | Existing directional tree-flag owners in both directions. |
| LessThanEqual, GreaterThan | Existing unequal-width tree and target shell. Source SDK allocation order is used for the comparison stream. |
| LinearDepthGreaterThan | Existing carry-chain owner, including the signed overflow/callback discrepancy. |
| Equals | Existing typed Xor/MCX/Xor owner. Physical dtype admission remains with the register binder. |
| CLinearDepthGreaterThan | Existing signed/zero extension, out-of-place sum, MCX and cleanup with both numeric binary control polarities. |
| Four LinearDepthHalf relations and private helper | Existing difference/flag ownership, inclusive X shell, exact measured erasure and source physical aliases. |
| SignExtend, SignTruncate | Existing signed and ones-complement prefix owner plus shared fanout. Symbolic declarations preserve the source classification of even a SymPy numeric width difference as symbolic. Original constructors also accept same-class unsigned/opaque metadata despite their signed annotations. Those permissive cases remain explicitly unsupported by this signed owner and these two identifiers remain partial. |
| MultiControlX, MultiControlZ | Existing predicate-controlled Pauli owner, including empty controls and all literal positive/negative patterns. Deprecated arbitrary-target MultiControlPauli remains open. |
| ControlledViaAnd dependency | Exact self-contained scalar X/Z child plus one nonempty QBit clause. Child identity is checked against the existing fixed-gate constructor, never inferred from its name. Scalar and array shapes retain source control order. Generic children and non-QBit/multiple clauses remain open. |
| ZGate, CZ | Existing fixed primitive leaf owner with q and q1/q2 signatures. This closes these two dependencies, without claiming the complete z_basis.py state/effect family. |

The fourteen comparison identifiers plus two Pauli specializations and two fixed
leaves bring complete built-in coverage from 42 to 60 of 303 identifiers. The
remaining 243 include the two signed-only conversion mappings, generic wrappers,
partial OnEach and generic ControlledViaAnd. Q-0876 remains open. Rejected source
symbolic HasLength serialization is preserved, while depth-zero symbolic roots
can be declared. The reference codec is unchanged.

## Execution and resource evidence

Named libraries and directly decomposed source libraries resolve through the same
existing component owners. Tests compare complete primitive dependencies, wire
operands, event-specific classical slots, resets and feedback. Unequal-width SDK
tree allocation uses its source scheduling witness, independently from the old
Cirq generator witness. All fifteen original example factories retain their
serialized signatures. Small complete source tensors cover complex amplitudes
and dirty targets with a spectator, every raw measurement history and zero work.
Directional flags and half-comparison images retain exact coherent round trips.
Ordinary append and trajectory execution are exercised, with accepted wider
complex64/128 and source model evidence reused.

| Case | Actual primitive inventory | Independent source model |
|---|---|---|
| n-control X/Z, n >= 1 | 4(n-1) T/TDG, n-1 measurements and resets, n-1 clean predicate wires | Same total T and measurements. Clifford/And categories remain separately queried. |
| Empty-control X/Z | One X or Z, no work or measurements | One Clifford. |
| Sign extension by k bits, either direction | 2k-1 CX, no T or measurements | MultiTargetCNOT is one atomic model Clifford. |
| Half comparison on n-bit words | n forward And calls, 4n T/TDG; reverse n measurements and resets | Same total T/measurement. Source difference/flag outputs stay distinct from private temporary work. |
| Other comparison families | Full emitted instruction streams checked against unmodified source | Existing independently accepted QEC categories and peak models reused from COMPARISON_FILE_AUDIT.md. |

The original comparison, conversion, multi-control and z-basis tests run unchanged.
Notebook cases are explicitly excluded from this fresh source selection. Complete
comparison notebook evidence is reused from the accepted 673-case notebook lane.
The source retains symbolic/model skips, strict nominal-dtype skips and expected
failures. No skip is described as a passed numerical check.

## Remaining gaps

Generic Adjoint/Controlled/Power, arbitrary-target Pauli and full structured
ControlledViaAnd/CtrlSpecAnd constructor closure remain separate work. Source
GreaterThanConstant has no numerical body. Signed linear overflow, strict dtype
mismatches, ge/lt aliases, source HasLength wire failure and model granularity stay
visible. Signed-only SignExtend/SignTruncate adaptation is partial for the source's
permissive same-class non-signed constructor inputs. Pending Proposals 116-119 and
whole-repository/GPU/performance acceptance remain open. This batch changes no
protected storage or numerical admission contract.

Batch receipts use `ftqc-reference-env/g2-comparison-constructor-*`.

## Batch verification

The constructor selection passes 253 cases with 100 percent statements and
branches across the four affected format owners. All eight changed Python files
pass strict typing, Black and Ruff. Compiler scope, repository language,
architecture, multilevel IR and protected API checks pass. The dependency checker
retains exit one with findings identical to the existing 720-line baseline.
Affected local integration passes 145 cases.

The four unchanged original source test files pass 604 cases, with 31 original
skips, nine expected failures and five notebook cases deselected. Eleven frozen
blobs and the real Kickmix 0.1 native extension with callable Simulator are
verified. The accepted 673-case comparison notebook lane and full component
source evidence are reused. The concentrated comparison/dependency/arithmetic
regression passes 2432 cases. These bounded local checks
leave whole-repository and accelerator acceptance open.
