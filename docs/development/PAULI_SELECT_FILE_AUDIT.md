# Pauli SELECT source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `select_pauli_lcu.ipynb` | `6513d5faf7b227cd26534f30e05a7d3b55a29f5a` | Q-0591 |
| `select_pauli_lcu.py` | `41e2f8ddb341fbbdd2d004ef533041bd24a5898c` | Q-0592 |
| `select_pauli_lcu_test.py` | `81ccdd543c60783cecee6a87ae238b050ed3b28b` | Q-0593 |

The [native tutorial](../../examples/notebooks/select_pauli_lcu.ipynb)
retains the four-row example, typed register signature, and all eight
direct call identities. It appends the bounded SELECT in uncontrolled,
zero-controlled, and one-controlled forms. Wires are ordered control,
big-endian selection, target in Pauli-letter order, then caller-owned
clean work. Measurement slots are explicit and the operation is
self-adjoint on the executable signed Pauli rows.

## Original grids and resources

The four-row tutorial suite checks the source gate dependencies,
operands and feedback in all three control configurations. A coherent
input spans every address, control state and target basis state. All
four uncontrolled and all eight controlled measurement histories are
projected directly, with full complex amplitudes and clean work checked.

The original four-site Ising cases cover all eight rows and both control
polarities, including inactive branches, using three raw histories for
each polarity. The three-site check compares the full output vector and
all eight original eigenstate overlaps against the source Hamiltonian
and preparation coefficients. The frozen source tests and notebook
yielded eleven passes plus the expected example serialization failure.

| Example | Emitted T/T† | Emitted CX | Emitted measurements | Source And / Cliffords | Source peak wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncontrolled | 8 | 17 | 2 | 2 / 18 | 7 |
| Active on zero | 12 | 24 | 3 | 3 / 20 | 9 |
| Active on one | 12 | 24 | 3 | 3 / 18 | 9 |

The declared model corresponds to 8 or 12 T equivalents at four T per
And. The source Clifford inventory is recorded separately from the
emitted CX and other physical instructions.

## Remaining source-file boundaries

Q-0591 and Q-0593 are accepted. Q-0592 now binds concrete foreign
dense-Pauli rows and serializes the resulting native logical and primitive IR.
Its source nonunitary coefficient domain and unresolved symbolic dimensions
remain explicitly outside the physical execution path. The locked source applies
the sign of each coefficient's real part. Nonzero real parts of either
sign map to signed unitary rows, regardless of coefficient magnitude
or imaginary part. This includes infinite real parts and discarded nonfinite
imaginary metadata. NaN real parts, zero and pure-imaginary coefficients produce a
nonunitary controlled leaf in the source. The native circuit accepts
explicit signed Pauli rows.

The bounded path preserves the source's padded-address folding. The
ordinary native logical SELECT continues to apply identity on unused
addresses. Existing differential tests preserve the padded address
`11` with a three-row table and its exact complex phase.

## Complete external row binding

The complete locked implementation, original tests, example factory and direct
unary traversal were read. The Ecosystem-owned `select_pauli_lcu_from_cirq`
adapter consumes the external fields and converts every `DensePauliString`
immediately to a native signed Pauli word. Algorithms retains its existing
`SelectPauliLCU`; Compiler retains bounded traversal and resource ownership.
Cirq is imported lazily in the adapter and no external row is stored in native
IR. The adapter preserves the source real-part sign rule, with nonzero
real parts of either sign, including infinite magnitude and discarded nonfinite
imaginary parts. Zero-real-part and NaN-real-part rows are refused as nonunitary.
Invalid widths, target lengths,
control values, insufficient address bits, empty tables, foreign row types and
NaN real coefficients are refused before construction.

The existing four-row factory, Ising grids, source dependency comparisons,
operand and feedback checks, resource comparisons and all small tutorial
measurement histories now consume the actual adapter. Additional signed
I/X/Y/Z rows with complex coefficients test every operator column and every
measurement branch for no control and both control polarities. Full complex
amplitudes include the raw branch normalization and work bits must return to
zero. Logical IR and primitive IR each survive JSON serialization. These cases
reuse the existing append entry and the original self-adjoint source semantics.
The ordinary logical unused-address identity and source bounded padded-address
folding remain separate, already verified behaviors.

The resource table above is unchanged because the adapter changes input
representation, leaving the traversal and emitter intact. Its emitted primitive
inventory and independent locked And/Clifford model remain separate evidence.
Symbolic widths require explicit concrete binding. The original nonunitary
zero/pure-imaginary coefficient leaves remain a recorded source boundary, not
an accepted native unitary operation.

## Complete coefficient normalization dependency

Both Ecosystem conversion paths now use `_pauli_select_rows.py` for the exact
sign-of-real conversion. Actual untouched nth_operation leaves prove that an
infinite nonzero real coefficient still becomes a signed unitary Pauli, and
nonfinite imaginary metadata is discarded. The old public adapter rejected
all nonfinite metadata while the stored-Program capture rejected infinite real
parts only. Neither restriction followed the complete original conversion.
NaN real parts and zero real parts remain nonunitary source leaves and retain
their native refusals. No external metadata reaches the resulting signed words.

Twelve finite-coefficient operations and twenty-four complete directional
Programs remain exactly equal to published b6da4d703. Newly admitted inputs
produce the identical complete Programs of the corresponding normalized rows,
so unchanged traversal, call, feedback and work evidence is reused. A four-row
mixed bank additionally compares actual source primitive dependencies, ordinary
append, independent resource categories, complete complex operators and every
small raw measurement history in both directions and all three control modes.

| Mixed IX / -XY / YZ / -ZI bank | Actual T-family / CX / measurement instructions | Independent source And / Clifford / measurements / logical peak |
|---|---|---|
| No external control, either direction | 8 / 16 / 2 | 2 / 14 / 2 / 5 |
| Active on zero, either direction | 12 / 23 / 3 | 3 / 16 / 3 / 7 |
| Active on one, either direction | 12 / 23 / 3 | 3 / 14 / 3 / 7 |

This closes the actual coefficient-admission discrepancy without adding a
primitive or changing a protected Core contract. Source nonunitary leaves,
symbolic constructor failures and object-array serialization remain separate;
this dependency closure does not by itself accept the whole-file parent.

Final affected regression passes 63 cases in each Cirq 1.7/1.6 profile without
skips, plus 145 local integration cases. Strict typing covers five Python files;
format, lint, owner scopes, language/links, architecture, IR and API checks pass.
Optional-test imports are guarded; frozen source and actual Kickmix availability
are asserted on the reference lane. Dependency policy retains 724 repository
findings, none on this batch.

The complete constructor-domain intake additionally identifies distinct work:
Boolean width/control metadata and an empty target execute in the original
root graph and generator. The scalar metadata follow-up below admits the
Boolean cases; empty target execution is covered by approved Proposal 129 below.
Concrete SymPy integers execute in the source generator while its stored graph
classifies them as parameterized. NumPy integer widths and floating widths fail
in the original QAny validator. Empty selection fails in the original traversal.
These protocol distinctions are recorded in
g2-pauli-constructor-domain-intake-20261126.json in the reference evidence
directory. They are not repaired by silently casting every numeric input, and
the source's out-of-domain control value two does not become a new native
binary-control contract.

## Complete scalar constructor metadata dependency

Both conversion paths now normalize Python integer/Boolean widths and concrete
SymPy integer widths at the Ecosystem boundary. Existing concrete binary
activation validation owns control normalization, including Boolean, NumPy
integer/Boolean and binary floating controls. All values entering native Programs
and structured instructions are ordinary Python integers; Core's strict numeric
metadata rules remain unchanged. NumPy and floating widths retain refusal.

The actual source generator's calls, operands and measurement feedback compare
against normal append for every accepted profile and both control polarities.
The complete directional Programs equal their canonical-integer counterparts.
Full complex operators, every small raw measurement history and clean work are
checked through those Programs in both directions. Twelve old operations and
twenty-four complete Programs still equal the previously published baseline.

Concrete SymPy integer widths retain two distinct source failures: stored graph
decomposition reports a parameterized gate, and QEC counting raises AttributeError
on Integer.bit_length. Both are explicitly reproduced. Its executable source SDK
generator is compared independently; native canonical resource estimates are not
reported as successful original symbolic-width model queries.

The affected batch passes 80 cases in each Cirq profile and 145 local integration
cases. Five touched Python files pass strict typing, format, lint and scope;
English/links, architecture, IR and API checks pass. Repository dependency policy
retains 724 findings, none on this batch. No frozen reference, primitive emitter,
allocation lifetime or protected Core contract changes. Empty-target admission
was completed by the approved empty-target follow-up below. Wider symbolic
declaration/control/capture scope remains separate work; Q-0592 stays open.

## Approved empty-target execution and shared reference support

[Proposal 129](API_CHANGE_PROPOSAL_129_EMPTY_PAULI_SELECT_TARGET.md) admits
zero-target signed rows through existing Core instructions, ordinary append
and typed Programs. Empty words retain their selected relative phase. The
typed target handle returns unchanged with no physical carrier. Majorana
positive-target and zero-wire rules retain explicit negative tests.

Fifteen actual source table/control families cover two-bit mixed/padded tables
and one-bit mixed, constant-positive and constant-negative tables. Both exact
directions compare complete source call layers and every unnormalized operator
column in every small raw measurement branch. Thirty independent source QEC
and peak rows agree; private work returns clean with no live leases. The
three-row table with a negative final row explicitly distinguishes original
bounded padding from logical unused-address identity. Dense, dynamic unary
and unitary-ladder logical paths retain their existing respective semantics.

| Empty / negative-empty two-row table | Actual primitives | Separate source model |
|---|---|---|
| No external control | 2 X, 1 Z; no work or measurement | 3 Cliffords, logical peak 1 |
| Positive control | 4 T-family, 7 CX, 1 measurement; one private bit | 1 And, 3 Cliffords, 1 measurement, peak 3 |
| Negative control | Same inventory plus 2 X | 1 And, 5 Cliffords, 1 measurement, peak 3 |

Source fixtures and generator comparisons now belong to
tests/unit/pauli_select_reference.py. All four existing consumer test modules
use that owner directly. Fixture and compiler function ASTs equal the published
source bodies except for the descriptive compiler-helper name; no assertions
or frozen source hashes were dropped. Other general test-harness dependencies
remain outside this domain extraction.

Affected regression passes 143 cases in each source profile and 145 local
integration cases; 79 existing Core IR/operand checks pass. Strict typing
covers nine Python files. Complete positive-target baseline Programs remain
identical. Original failure logs and proposal intake remain separate from
successful final receipts. Contextless logical Programs still have no public
native-physical codec; their internal object identity round trip and actual
logical/primitive JSON round trips are recorded distinctly.

## Owned symbolic declaration and concrete materialization

The direct constructor boundary retains the complete signed row tuple and the
original symbolic selection expression, including integer and positivity
assumptions. Algorithms owns this immutable declaration; Ecosystem converts
external rows immediately. The existing neutral `SymbolicInt` vocabulary is
reused. No external Pauli object, source callback, declaration-only empty
Program, new public entry or Core physical symbolic width is introduced.

Selection remains MSB first, targets remain in Pauli-letter order, and controls
retain their optional binary activation. Logical output registers are unchanged.
The declaration owns no physical workspace. After an explicit immutable
`dataclasses.replace(operation, selection_bitsize=width)`, existing bounded
append and typed capture own unary private work and measured cleanup exactly
as in concrete source calls. Unbound physical append and Program capture refuse
execution. Ambiguous capacity comparisons retain the original constructor's
symbol-assumption refusal instead of inferring a concrete dimension.

Positive integer `n` with two rows and `n+1` with three rows construct original
signatures while original stored decomposition refuses them as parameterized.
Materialization covers each minimum width and one extra bit, no control and
both control polarities. Both directional Programs retain complete source call
layers, operands, feedback and output ownership. Every operator column in every
small raw measurement history compares full complex amplitudes and clean work.
Normal bounded append and numerical branch execution run with source SDK
imports blocked. Resource measurements and independent source estimates remain
separate. Internal declaration pickle identity does not claim a public Program
codec; original object-array serialization and generic control remain distinct
whole-file gaps. Q-0592 remains open until those dispositions are complete.

| Materialized three-row -X / Y / -Z table, width 2 | Actual T-family / CX / measurements / private peak | Separate source And / Cliffords / measurements / logical peak |
|---|---|---|
| No external control, either direction | 4 / 8 / 1 / 1 | 1 / 9 / 1 / 4 |
| Active on zero, either direction | 8 / 15 / 2 / 2 | 2 / 11 / 2 / 6 |
| Active on one, either direction | 8 / 15 / 2 / 2 | 2 / 9 / 2 / 6 |

All twenty-four materialized directional inventories match their independent
source models and peaks; no private lease remains at return. This is concrete
execution evidence, not an assertion that the original unresolved symbol has
a successful physical graph or numerical resource query. The branch-operator
checker now has one shared Pauli reference owner used by both empty-target
and symbolic-materialization consumers; its assertions are preserved.

Affected regression passes 150 cases in each Cirq 1.7/1.6 profile without
skips, plus 145 local integration cases. Seven touched Python files pass strict
typing, format, lint and owner scopes; language, links, architecture, IR and API
checks pass. The final focused materialization check passes after import-guard
cleanup. Dependency policy retains 724 repository findings and none on the
touched paths. Twelve old operations and twenty-four complete directional
Programs still equal the published baseline.

## Control wrapper interface and ownership

The existing Ecosystem capture entry owns source wrappers as well as explicit
Pauli SELECT objects. Specialized controls retain their source activation
array order, with each newly added control preceding the old one. Structured
control words retain original register names, shapes, dtypes and MSB bit order.
All ports are THRU; targets and controls return to their caller. Compiler's
existing specialized-control and predicate owners hold temporary conjunction
bits through the positive Pauli child and measurement-erase them before return.
No foreign object or source callback is retained by the returned Programs.

An explicit generic Adjoint wrapper reverses actual typed calls. A child's own
source adjoint specialization is honored separately: Pauli SELECT returns
itself, which is not a request to reverse its measured primitive stream.
Predicate compute and erasure use their explicit directional definitions.
Original source callback failures and object-array codec failures remain
separate from the native executable wrapper body.

The capture now handles original specialized multi-control, structured
ControlledViaAnd, nested wrappers, and explicit Adjoint objects using existing
Compiler owners. Scalar, singleton, shaped and independent control registers
and unsigned two-bit activation words are included. Fifty-four actual source
table/control/target families compare both directional root graphs, named
operands and feedback. The fully flattened original typed graph supplies the
primitive reference, with each fixture's one measurement per private bit bound
to the corresponding native SSA result slot. Per-wire gate dependency streams
match without sorting away shared-wire order. Every operator column in every
small raw branch matches source complex amplitudes, with private work clean.
Source-free binding, normal append and branch execution also pass.

This comparison exposed and fixed an existing native inverse leaf discrepancy.
The source controlled dense Pauli adjoint is the same controlled Pauli leaf.
The old native inverse instead reversed its primitive letters and sign gate.
Both streams had the same operator, but differed in shared-control gate order.
The Compiler now retains the original leaf order inside the reversed outer
call graph. Twelve previous operation objects and complete forward Programs
are identical; twelve inverse Program differences are confined to the Pauli
inverse-leaf definitions. Their primitive inventories, output bindings and
allocation lifetimes are unchanged. Earlier operator/call checks alone did
not prove this primitive order; the new full-stream evidence supersedes that
overly broad interpretation.

The 108 separate directional resource inventories match shared native
predicate/positive-child estimates, original QEC categories and original
peak-width queries. No private lease remains at return.

| Signed X / Y bank with existing positive control | Actual T-family / CX / measurements / private peak | Separate source And / Cliffords / measurements / logical peak |
|---|---|---|
| Add two-bit array activation (1, 0), either direction | 12 / 20 / 3 / 3 | 3 / 7 / 3 / 8 |
| Add nested zero and two-bit array controls, either direction | 16 / 26 / 4 / 4 | 4 / 8 / 4 / 10 |
| Extend specialized helper with a third control, either direction | 12 / 20 / 3 / 3 | 3 / 11 / 3 / 8 |

The raw SDK wrapper's inherited decompose_from_registers is not a physical
generator. Expanding deferred SDK children after graph export also allows the
test manager to recycle an outer predicate too early. Those failed probes are
retained separately; the accepted comparison uses the original fully flattened
typed graph, which carries all lifetimes explicitly, without changing source.

Original nonempty Pauli rows still fail object-array serialization. Empty
targets serialize, but deserialization either misses the original private
specialized-helper constructor or returns NumPy row arrays: equality then raises
an ambiguous-empty-array ValueError and execution raises missing ndarray.on.
All 27 empty-target encoding cases retain distinct serialize, deserialize,
equality and tensor outcomes. A further wrapper around an already-specialized
helper reaches the original np.concatenate callback TypeError; native capture
retains explicit refusal rather than repairing the reference. These are
source-file/dependency boundaries, not newly missing inverse bodies. Existing
nonunitary coefficient and out-of-domain control boundaries remain recorded.
Q-0592 stays open under the current whole-file acceptance rules; the approved
three-parent defect disposition does not apply to it.

The final affected regression passes 207 cases plus the separately verified
callback counterexample on Cirq 1.7, and 208 cases on Cirq 1.6. Local integration
passes 145 cases. Three Python files pass strict typing, formatting, lint and
owner scopes; language/links, architecture, IR, API and diff checks pass.
Dependency policy retains 724 repository findings with none on touched paths.
Original logs include the failed setup/probe paths and the final accepted
flattened reference; only final receipts support these acceptance statements.
