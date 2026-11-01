# Bounded lexical scopes inside apply

Extend #993 only after its protected merge. An apply block may contain another
supported within/apply scope. Preparation blocks still admit only fresh local
quantum-expression assignments; nesting there remains refused. Function-level
declarations, contextual identifiers, Const-only actions and phase restrictions
remain unchanged. Limit lexical scope nesting to the existing MAX_DEPTH (32),
independently of the existing expression-depth and expansion limits.

Each scope tracks only its newly materialized owned locals. Exiting an inner
scope uncomputes/releases them in reverse preparation order and restores the
parent scope context. Outer locals remain live and may feed inner preparations.
Reinitializing a live outer local remains a Compiler initialization error;
inner locals cannot escape. Sequential inner scopes may reuse released source
names with their fixed widths and new identities. No shadowing or block-local
declarations are admitted.

Reuse Compiler verification, selected exact inverses, local pooling, whole-width
resource selection and Program/run/named results. No new canonical IR, Compiler
operation, public API, role validator, executor or dependency. Admit only the
existing unsigned compute/phase/cleanup profile, not general unitary/control,
Input/free/drop or arbitrary coefficients.

Acceptance must compare full coherent amplitudes, reference entanglement and
hidden zero work under both objectives, all sixteen two-bit inputs, helper versus
lexical composition, sibling reuse, depth/expansion caps, parent lifetime,
escaped/uninitialized references and actual budgets. Original source/license
digests and four-original cohort remain unchanged. Independent trials remain
0/2 and release_authorized=false. Test counts and resources require execution.
