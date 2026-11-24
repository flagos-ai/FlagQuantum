# LinearCombination repeated control and exact specialization closure

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
This completes the remaining source-supported control recipes for the scoped
LinearCombination constructor, following the default oracle closure in
[the previous batch](DEFAULT_LINEAR_COMBINATION_ORACLES_AUDIT.md).

## Interface and ownership

The source already-controlled LCU has scalar positive `ctrl`. Adding another
single control produces `_MultiControlledFromSinglyControlled`, with a QBit
array in the order `(added activation, existing activation)`. Further concrete
helper declarations can extend that array. Grouped controls use the original
ControlledViaAnd reduction before this two-control specialization. All data and
existing control storage remain caller-owned; predicate scratch starts and ends
zero and is measurement-erased by the shared predicate owner.

Compiler's existing `multi_controlled_component_program` remains the authority
for forward composition. Its new exact pair owner reverses logical helper
calls using explicit positive-child inverse and compute/erase pairs. It never
reverses measured primitive instructions. Formats adds a scoped constructor
recipe for the original helper, validates the actual `ctrl_bloq`, binary cvs,
control port and native signature, and retains exact generic inverse recipes.
The existing ControlledViaAnd factory receives the source specialization adder
instead of rejecting an already-controlled LCU. No new Core opcode, public
operation, backend selection or external object enters this boundary.

The shared test binder now flattens row-major register elements rather than
treating a shaped control array as a single word. This fixes its test-only
physical binding convention without altering production storage admission.

The broad shaped-handoff regression also caught a pre-existing admission-order
defect: a symbolic array declaration reported missing child body before its
nonmaterializable shape. An isolated interpreter executing the exact parent
module reproduced the same failure. The shared helper now asks the existing
Register shape validator to materialize dimensions before traversing the child,
matching the source finite-builder order. The original failure assertion is
unchanged; symbolic word widths retain their separate existing declaration
behavior.

## Complete-file evidence and retained source boundaries

The original default signed four-block example, alias PREPARE, packed SELECT,
positive controls, generic inverses, complete small operators, nonuniform
eight-bit source table, sparse complex roundtrip and SDK-free ordinary append
were accepted in the prior batch and are reused. Existing scoped context
evidence covers unequal child widths, explicit custom/measurement-bearing
oracles, nominal types, symbolic declarations, unused precision and lossy
payload ambiguity. Native providers remain explicit for arbitrary child types;
this is a constructor boundary, not an inference from a register signature.

This batch compares the actual repeated-control calls, operands and outputs
for zero/one scalar additions, two-/three-bit grouped additions and three-/four-
element specialization arrays in both generic directions. Full complex logical
operators and final work zero are checked. The small repeated-control cases
also numerically enumerate every raw measurement history on a complex input,
including its branch normalization. Default-source capture, scoped resolution,
binding and `circuit.append(op)` additionally run with source imports disabled.
Actual primitive/lifetime resource receipts remain separate from unchanged
source QEC and logical-peak estimates.

Batch acceptance: 385 concentrated cases and 145 local integration cases pass;
four affected source/test files pass strict typing, Black and Ruff. Team scope,
repository language, architecture, IR/API and whitespace checks pass. Dependency
findings match the unchanged 720-line baseline. The eight directional resource
rows in `g2-repeated-lcu-control-resources-20261125.json` show the scalar-zero
addition at eight T-family, one atomic CCX, twelve CX, two measurements and
two private/seven live wires. The source separately reports one Toffoli, two
And, eight Clifford, one rotation, two measurements and peak seven. A grouped
010 addition emits sixteen T-family, one atomic CCX, twenty-four CX and four
measurements, four private/eleven live wires, versus one Toffoli, four And,
eighteen Clifford, one rotation, four measurements and peak eleven. Neither
resource equality nor cleanup is inferred merely from lease release.

The original helper's further-control **callback** has a distinct upstream
failure: `np.concatenate(ctrl_soqs, in_soqs[ctrl_reg_name])` passes its old array
as the axis argument. The unchanged callback and grouped-control decomposition
both raise `TypeError` identifying integer scalar arrays. Native scoped
resolution refuses that unsupported outer owner rather than silently repairing
the source callback. Direct extended helper declarations still decompose and
execute successfully and are tested independently. This is not a reason to
reject the working repeated-control LCU routes above.

Original shallow private-payload omission, deeper object/ragged tuple codecs
and specialized-adjoint enum serialization failures remain unchanged. Explicit
native constructor contexts and the supported generic adjoint wire retain their
documented roles. Catalogue acceptance describes the complete source-supported
constructor and its explicit native role boundary; it does not claim a repaired
upstream serializer or arbitrary automatic SDK child conversion.

## Next dependency inspection

Proposal 119 remains unapproved and is not implemented by this work. However,
the existing accepted `legacy_scalar_factories.py` already builds a registerless
GlobalPhase Program using an allocated/restored/freed scalar carrier. Before
classifying Phase, reflection or Chebyshev roots as blocked, the next batch must
test this existing recipe in their actual source call chains. Its extra physical
work and primitive count must be reported separately from the source's zero-cost
scalar model. Do not introduce the pending `scalar_phase` Program operation or
assume the earlier blanket scalar blocker still applies to these existing owners.

The intake receipt `g2-phase-existing-scalar-probe-20261125.json` already verifies
the unchanged original Phase(Hadamard) example's parent calls and full complex
operator using that existing scalar owner and fourteen native definitions,
including final private work zero. Therefore this concrete next module has a
working dependency path without a new Core operation. Broader Phase/reflection
constructor and exact inverse closure still needs its own full batch acceptance.
