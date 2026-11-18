# Complete comparison notebook audit

Q-0179 covers all 108 cells of the original comparison notebook at Qualtran
v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`7dfc96afc9eb98246acbb7a949d1f5b6325888b7`. The reference notebook executes
unchanged. The native notebook retains the section order and every instance,
with two additional cells exercising existing native append examples.

## Ownership and complete disposition

The fixed teaching catalogue lives in `examples/algorithms/comparison_catalog.py`.
It assembles existing Core signatures and uses the existing Compiler graph and
Drawer owners. Its full callee labels retain declared widths, types, constants,
control polarities and directions. Labels describe this finite teaching asset.
They introduce no executable foreign-object admission or new production model.
Symbolic signatures allocate no physical state. The native kernel denies both
Qualtran and Cirq imports.

| Source section | Native disposition and existing direct dependency |
| --- | --- |
| GreaterThan, four-bit words | Original LessThanEqual(4,4) and X edges. Physical `ordering` composes the existing measured comparison tree. |
| GreaterThanConstant, four bits and 13 | Original same-threshold LessThanConstant edge retained as a declaration. Source execution is absent. |
| EqualsAConstant, four bits and 13 | Original `(1,1,0,1)` MCX. Native equality owns the physical predicate and shared reduction. |
| LessThanConstant, symbolic n/k and eight bits/5 | Both signature instances, exact symbolic assumptions and concrete four-child graph. Shared `less_than` owns gates, feedback and workspace. |
| BiQubitsMixer | Two unsigned two-bit inputs and a RIGHT opaque three-bit history. Its X, nine CNOT and two And declarations retain direction. |
| SingleQubitCompare | THRU a/b and RIGHT less/greater flags. Its X, four CNOT and one And graph is distinct from the mixer. |
| LessThanEqual, symbolic n1/n2 and concrete 4/8 | Both signatures and every concrete prefix/tree/erasure edge. Same-named mixer and single-comparison adjoints remain separate identities. |
| Controlled linear comparison, signed five-bit words | THRU ctrl/a/b/target, two typed complements, full adder/adjoint, positive MCX and both sign extension/truncation calls. |
| Equals, unsigned four-bit words | Two typed XOR calls and all-zero four-control MCX. |
| Four unsigned three-bit half comparisons | RIGHT four-bit difference and target. Each preserves three complements, the truncated widened adder and CNOT. Inclusive relations add X. |
| Drawing and execution | All 13 signature groups and 13 one-level graphs use native Drawer. Existing `register_comparison` and `register_half_comparison` examples use `circuit.append(op)` and verify complete complex states and clean work. |

The original source has 15 instances, including two symbolic-only instances.
Every original port name, dtype encoding, width, shape and side is checked.
Every original one-level callee identity and count is checked with all its
constructor fields, including the two distinct adjoint tree nodes. Existing
component suites retain exact gate/operand/feedback comparisons and applicable
small measurement-branch certificates. This batch reuses those suites.

## Separate physical and model accounts

| Example | Emitted primitive account | Independent source model |
| --- | --- | --- |
| Three-bit x<5 append | 12 T/TDG, 26 CX, 3 measurements, 4 clean work | Existing constant-comparison model and all exact gates remain covered. |
| Three-bit x<8 append | One X, no work or measurements | Boundary model is one Clifford. |
| Two-bit half compute/copy/erase, unsigned | 8 T/TDG, 39 CX, 2 measurements, 6 temporary wires zero | Compute/erase model 8 T equivalents and 2 measurements. Result copy is an additional CNOT. |
| Two-bit half compute/copy/erase, signed | 8 T/TDG, 47 CX, 2 measurements, 6 temporary wires zero | Sign extension costs remain in the independently queried model. |
| Eight-bit x<5 model-only notebook query | No eight-bit state allocation | 32 T equivalents, 8 measurements, 18 peak logical wires. |
| Three-bit half model-only notebook query | No three-bit physical emission in that cell | 3 And, 30 Clifford, 12 peak logical wires per forward call. |

Both target initial values and coherent spectators remain covered by the
existing component suites. The append demonstrations exercise both signednesses
and all four half relations, and constant comparison applied twice restores its
input. Raw feedback and logical output aliases retain their distinct contracts.

## Source and admission gaps retained

The original GreaterThanConstant graph points to less-than at the same threshold,
and its root supplies neither executable decomposition nor classical action.
The native asset reports that model without claiming execution of greater-than.
The source teaching equation used an AND output, which is inconsistent with the
usual reversible XOR target. Native prose states XOR and records this correction.
The half-greater-or-equal and half-less-than register prose is corrected to
a-b, consistent with their input exchange in the actual component body.
The reference notebook is unchanged.

Half-comparison adjoint output aliases can exchange a/b. The existing source
alias and preserving-physical-order routes remain separately documented. Signed
symbolic SignExtend peak queries and generic symbolic/foreign execution admission
remain open under their implementation parents. Q-0179 is a teaching-asset
closure and does not close those implementation gaps.

The existing signature renderer only inspects the first two tuple entries. Its
type annotation now accepts opaque graph metadata, allowing comparison and field
call labels to reuse the same drawing function without executable coupling.

Batch acceptance passed 673 concentrated reference/native cases with no skips.
The existing IPython temporary-directory warning remains recorded. Changed
Python and extracted notebook code passed strict mypy, Black and Ruff. Compiler
scope, architecture, multilevel IR and the protected API baseline passed.
All 26 figures were rendered and inspected, with wide call trees arranged
left-to-right to keep complete callee identities readable. The four finite-field
teaching consumers of the reused signature renderer passed in the same batch.
