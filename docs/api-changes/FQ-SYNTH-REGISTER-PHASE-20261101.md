# Ordered predicate phase calls and basis-constant borrows

Date: 2026-11-01. The human instructed merging #982 and continuing the sixth
plan. Exact head cc6d69bad7d4617c7473b203580db424add3d203 passed 57 checks
with three configured skips; merge is 8577d6e3967bbd017ff57e0063915cfa5ffbe90e.
This stage-2 dependency slice gives live register composition an audited phase
borrow action, before general quantum function roles and resource scopes.

## Existing roles, new action

Existing qreg.declare/allocate/initialize/compute retain their fresh-output and
initialization rules. qreg.compute borrows immutable basis values and writes a
fresh destination. Existing arithmetic value calls have exact XOR semantics;
they do not accept arbitrary external unitary implementations or effect claims.

Add private `qreg.phase(effect, *registers) -> next_effect`, with exactly one
`callee` attribute and no regions. The callee is an existing verified unsigned
expression function with an exactly one-bit return. It applies (-1)**f to the
joint basis encoding of initialized borrowed registers and returns each
reference still initialized with the same basis value. It may change relative
or overall phase and is therefore a basis-constant borrow, not an unchanged
state, a classical constant or a consumed Input. Output initialization remains
separate. No user-asserted Const/effect annotation can bypass verification.

Arguments must be live declared initialized register references with exact
formal widths. Repeated actual aliases are legal immutable borrows; lower
through the existing unique-argument call wrapper, never cloned states or
overlapping physical entry bindings. Unknown callees, wrong roles/types/widths,
non-one-bit results, extra attributes, nested regions and stale effect tokens
fail at verification. No free/drop, consuming input, mutable external borrow,
higher-order arbitrary unitary, generic control or within-apply syntax is added.

## Resource scope and reuse

The existing arithmetic scheduler builds the predicate prefix once, marks its
live one-bit result (or direct product), and emits the exact inverse of the
same selected prefix. Every internal work qubit returns to zero before the
ordered action returns. The source may then mutate a borrowed register or reuse
that zero tail in a later compute/phase call. No dependency can mutate inside
the action because the callee is the closed immutable expression profile.
This is proof by construction at one call boundary, not proof of arbitrary
mutable within/apply bodies and not a free assertion about external state.

Reuse HybridProgram, quantum.register/uint, the linear quantum.effect, existing
function verifier/lifetimes/kernels, retain/recompute, RegisterCandidate segment
records, finite candidate/target selection, existing checked Program/run/read.
Share the existing phase target admission check: native logical or unoptimized
PHASE_BASIS (with admitted routing); reject projective target conversions.
There is no new type, contract object, report field, serializer or executor.
All production changes stay in Compiler. Native Qmod/Python public grammar and
public signatures/snapshots remain unchanged; private producers demonstrate
composition through the existing Services bridge. Public admission is a later
review after this role/cleanup dependency is established.

## Acceptance

Verify initialization/effect/callee/width/one-bit/alias rules and unsupported
claims. Compare complete complex states, external entanglement, constant signs,
phase actions separated by mutations, and phase/compute workspace reuse under
both schedules. Preserve global phase; prove workspace zero and effect ordering.
Exercise exact target conversion/routing, named results and plan identity,
budgets/memory/source errors before planning, full existing arithmetic and
register behavior, and original-size Qmod acceptance. Record actual costs and a
runnable private Compiler contributor example alongside the unchanged public
high-level predicate path. No public role DSL or all-216 compatibility claim;
independent trials remain 0/2 and release_authorized remains false.


Observed target boundary: the interleaving fixture passes native logical and
PHASE_BASIS routing with the final physical layout retained. Its restored-layout
route is refused before planning because existing Runtime normalization cancels
restore SWAPs and the target's witness audit no longer matches. Preserve that
fail-closed refusal and add a negative test; do not weaken layout evidence or
claim restored routing for this fixture. General optimizer/routing evidence
repair is separate from this call-role slice.
