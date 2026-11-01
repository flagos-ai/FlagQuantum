# Bounded native within/apply source admission

Date: 2026-11-01. Continue sixth-plan source scopes after #992, merged at
a38588eb502e6f3465b5d81d1cc3387787d5dd33 with 57 successful checks and three
configured skips. Only existing Ecosystem syntax/translation production modules
change. Compiler owns initialization, Const effects, inverse selection, release,
alias/lifetime proof, resource accounting and targets, unchanged from #991/#992.

## Source profile

Admit function-level scalar local declarations `flag:qbit;`, `tmp:qnum<N>;` and
bare `tmp:qnum;`, followed by nonnested lexical
`within { preparations } apply { actions }` statements. `within` and `apply`
become keywords. Local declarations must precede use and cannot shadow parameters,
locals or intrinsics. An unused declaration owns no physical storage. Local
reads/initialization outside within/apply are refused.

The within block contains one or more fresh assignments to declared locals,
using admitted quantum-dependent unsigned arithmetic/comparisons. Earlier
prepared locals may feed later expressions; Compiler rejects uninitialized reads
and repeated initialization. Pure literal preparation, allocate/gates, Output
preparation, helper preparation calls, local declarations inside blocks and
lexically nested within/apply remain outside this slice. Fixed quantum-dependent
expressions such as x*0+1 retain their declared dependencies.

Apply contains existing phase(expression,pi) statements or positional calls to
Const-only helpers. Compiler verifies their effects independently; the adapter
does not add another role validator. Polynomial/one-bit/pi phase restrictions
remain, including refusal of direct comparison phase arguments. Arbitrary gates,
assignments, allocation, Output calls, control/invert and general mutable apply
effects are refused. Empty apply is allowed; empty within is refused. Helpers
may themselves contain supported scopes, so semantic call composition is allowed
while lexical nesting remains bounded out.

## Translation and scope

A local is materialized as qreg.local only when needed inside a scope. Assignments
emit existing qreg.compute. After apply, emit qreg.uncompute and qreg.release in
reverse preparation order. Reuse the same selected inverse and zero-pool proof;
do not synthesize an alternative inverse or inspect a simulator state. Compiler
also checks malformed/unused helper definitions and the expanded complete root.

At scope exit the source variable becomes uninitialized/unbound again; a later
scope can reuse its name with a new owned SSA identity and internal label. An
explicit width is retained; bare qnum infers once and then keeps that width,
refusing later widening/truncation. Local widths do not leak into the output-only
inferred-width map or named results. Scalar source kinds still match helper
formals exactly. No implicit copy of an unknown quantum state is introduced.

The existing private AST stores declaration/block syntax, then is discarded.
No new canonical IR, Compiler operation, API type/export/signature/report field,
execution interface, dependency or SDK is introduced. Existing source bytes,
tokens, expression depth, function count, operation/node/visit/width and memory
budgets cover generated locals and cleanup. Diagnostics retain source coordinates.

## Acceptance and limits

Run all sixteen two-bit inputs, complex coherent states, external references,
global/relative signs, interference and zero hidden work under both schedules.
Verify chained local dependencies, repeated scope-name reuse, helper/nested-call
composition, Const aliases, full file/text input and replacement by the existing
Python/owned producer through unchanged Program/run/named reading. Check actual
hidden-work/qubit costs, complete budgets/target guards and no synthesis execution.
Refuse escaping/unprepared locals, illegal prepare/apply effects, malformed
blocks, unused invalid helpers, wrong widths/kinds and over-budget generation.

Update the former blanket local-declaration refusal to preserve refusal of
unscoped local initialization. Preserve original Qmod/manifest/license bytes and
their four-original execution cohort; new synthetic examples do not enlarge it.
This is a bounded compute/phase/cleanup source profile, not full Classiq within,
generic free/drop/Input, arbitrary-angle/control/unitary or all-216 compatibility.
Independent developer trials remain 0/2 and release_authorized=false.
