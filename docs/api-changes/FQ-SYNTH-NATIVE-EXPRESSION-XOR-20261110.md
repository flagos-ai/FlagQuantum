# Native expression XOR into initialized unsigned outputs

The authorized native example migration admits `target ^= expression` in a root
Python qfunc when target is an owned, initialized unsigned scalar Output and
expression uses the existing checked unsigned expression grammar. Its exact
unitary meaning is `|inputs, target> -> |inputs, target XOR expression(inputs)>`.
Expression results are zero extended to target width; truncation is rejected.
The target must be disjoint from every expression input, including canceled
reads. Inputs and target may be in arbitrary coherent states and need not be
zero. Existing literal raw-mask XOR behavior, including signed ports, is unchanged.

An internal `qreg.xor` ordered effect names the existing pure expression callee.
The verifier checks initialized mutable target, initialized inputs, disjointness,
matching expression input widths and a fitting result. The write invalidates
cleanup proofs that depend on the old target and never establishes a clean-zero
or fresh-computation proof for target. The effect is its own inverse while its
inputs are unchanged. Checked root adjoint/within regions may contain it; inverse
composition uses the same expression callee. The existing arithmetic/Boolean XOR
kernels and work scheduling remain authoritative. There is no new public export,
Python signature, physical IR opcode or numerical execution kernel.

The first admission excludes quantum arrays/subviews, borrowed mutable helpers,
Const targets, signed/scaled expression operands or targets, implicit truncation,
measurement, reset and target-dependent expressions. Unsupported cases fail closed.
Classical-only nonliteral masks retain the existing refusal; no host evaluation
is added. Existing node, expression, gate, depth, memory and work limits apply.

The first source path is the concrete expanded square-root QSVT instance in
`functions/qmod_library_reference/classiq_open_library/qsvt/qsvt.qmod`: comparator
XOR and its inverse can remain explicit high-level arithmetic. Projected RZ
phases must retain their half-angle and conditional global phase, and both
indicator and auxiliary outputs remain live quantum data.

Acceptance covers all small-domain inputs and nonzero targets, zero extension,
coherent complex phases, repeated XOR, adjoint and within cleanup, target alias
and stale cleanup rejection, initialized storage and format boundaries, actual
resource limits, independent Circuit comparison for the untouched QSVT original,
and preservation of all 216 frozen source identities and baseline resource rows.
This is an unreleased Compiler admission, not a release or complete Notebook claim.
