# Single-bit control metadata

This closes the one-bit metadata prerequisite from the Q-0158 controlled-source
intake. Frozen `CtrlSpec.get_single_ctrl_val` accepts one encoded bit across
concrete types and singleton shapes. Compiler now reuses native split/join to
feed a QBit to the positive control callback and reconstructs the original
word type and array shape afterward. Existing QBit scalar/vector paths keep
their original body. Core physical admission stays authoritative.

## Interface and source fidelity

The existing `control_component_via_predicate` signature is unchanged. It accepts
one QBit, QAny(1), QUInt(1) or QInt(1), with shape `()`, `(1,)` or `(1,1)`.
The signed activation -1 encodes one, and zero encodes zero. Negative controls
use the original X-control-X sequence. Non-QBit words use existing typed
split/join calls with zero physical gates. The original type and shape survive
both outputs and actual physical binding. Multiple empty and singleton control
registers remain refused by the single-register path.

This is a native wiring adaptation. The source permissive dtype comparison
allows a non-QBit one-bit word to connect directly to a QBit leaf. Native typed
linear handles use explicit split/join instead. Those bookkeeping calls are
separate from emitted primitive count and source call graph identity.

## Reference defect

The frozen ControlledViaAnd implementation extracts a vector element for shape
`(1,)` but leaves a `(1,1)` object array intact. Both positive and negative
controls then fail its builder scalar-Soquet assertion. Its standard tensor
contraction and Cirq unitary fallback also fail at that decomposition. The
source `_tensor_data()` method still supplies the intended matrix.

Tests preserve all three original failures for `(1,1)` and independently compare
native execution with that unchanged direct tensor method. Scalar and vector
cases compare original emitted gate operands as well as full tensors. This does
not claim source decomposition parity for the defective matrix shape.

## Execution and resource evidence

All 24 type/shape/polarity combinations retain the original signature, agree
with `get_single_ctrl_val`, and execute through `Circuit.append(op)` on a full
complex state with an entangled spectator. Repeating controlled X is its exact
inverse and restores the initial state. No work or measurement is introduced.
Positive control emits one CX. Negative control emits two X and one CX. Source
QEC counts agree for the original decomposable scalar/vector cases, including
zero T count. Matrix-shape source model closure is not inferred from these cases.

The focused run passes 102 cases, including existing structured controls.
Full concentrated regression, strict typing, format, scope, architecture and
IR results are recorded in `ftqc-reference-env/g2-one-bit-controls-*`.

The concentrated regression passes 3366 cases with one existing warning.
Strict typing passes three files. Black, Ruff, scope, architecture and IR checks
pass. Dependency scan reports 687 global findings and zero changed-path findings.
