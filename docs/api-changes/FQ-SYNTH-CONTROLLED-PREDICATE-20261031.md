# Direct controlled-predicate phase lowering

Date: 2026-10-31. The human instructed merging #976 and continuing the sixth
plan. #976 merged as 3ab96490d5de080e48f909cbceee517cb74db04b after its exact
head a6cb0e3427498208a63ea9f1daba1d1e955532b0 passed 57 checks with three
configured skips. This slice advances controlled-predicate construction behind
the existing expression/phase boundary; no new source/control API is introduced.
Independent trials and release permission remain pending.

## Effect and scope

For a one-bit control c and predicate f(x), the admitted expression
`return c and f(x)` in output_mode="phase" implements
|c,x,0_work> -> (-1)**(c*f(x)) |c,x,0_work>.
The control-zero branch is exact identity, including absolute phase. Input
preparation remains outside the oracle. A constant-true predicate produces Z on
the coherent control; it is not an unobservable phase that can be discarded.

When the terminal returned SSA operation is multiplication of two quantum.uint(1)
values, ensure both operands and mark their encodings directly with CZ. Equal
physical encodings reduce to Z because b*b=b. The multiplication result and its
local scratch are never materialized. This is valid for both explicit Boolean
AND and a one-bit numerical product. Other returns, classical/mixed/wide
products and nonterminal returned definitions retain the general phase path.
Value XOR is unchanged.

Reuse the existing lifetime analysis and retain/recompute scheduler. Keep both
operand encodings live at the marking boundary; ensure a dependency erased
under recompute is restored first. Snapshot the actual selected forward prefix
U, then emit CZ (or Z for aliases) and that literal prefix's adjoint. Do not
control each compute/uncompute gate, synthesize another inverse or add a new IR.
The existing X/CX/CCX prefix proof, operation/event budgets, binding/zero
preconditions, finite selection, target checks, Program/run and readout apply.

The original semantic graph remains the source of dependencies; allocation
traces record only executed work and no event for the omitted terminal product.
Peak workspace and operation counts include actual forward/inverse work. A
smaller peak is possible, not guaranteed when earlier predicate work dominates.
No generic Circuit.control legalization, within-apply block, arbitrary-angle
phase, routed public phase, dirty external workspace, type, opcode, signature,
export, report field, executor or serialization is admitted by this optimization.

## Acceptance

Check both schedules against an independent general selected-prefix baseline
extracted from the unchanged value evaluator's allocation trace. Compare complete
complex amplitudes, not only probabilities or phase quotients. Cover control-zero
and control-one branches, constant-true/false predicates, aliases, overlapping
logical reads, nested dependencies and external-reference entanglement. Verify
literal same-prefix inversion, exact resource differences, workspace zero,
constraint and operation boundaries, controlled interference and double application.
Public users keep one-bit controls in inputs and explicitly read their names.
Existing value, Boolean, phase, target and original-Qmod tests remain authoritative.
