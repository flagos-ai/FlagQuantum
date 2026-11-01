# Closed register calls with Const borrows and fresh outputs

Date: 2026-11-01. Continue after #986 (57 successful checks, three configured
skips; merge 819b813e61dd7b433d8014cd7d9d907766483f09). This Compiler-owned
dependency establishes initialized borrowing before public native Const syntax.
No public signatures, schemas, exports or source grammar change.

## Owned representation

Reuse HybridProgram, Block, quantum.effect and quantum.register. A register
callee has block arguments `(effect, *const_references)`, declares zero or more
fresh output references with existing qreg.declare, and returns the current
effect followed by every Const reference and declared output in that order.
Register entry arguments encode basis-constant borrowing in this closed profile;
they are not consuming Input or arbitrary mutable references. A complete root
program still accepts exactly one effect and requires all-zero allocated entry
storage. A borrowed callee cannot be executed as a fresh root.

Add private `qreg.call(effect, *const_actuals, *output_actuals) -> effect` with
exactly a callee attribute and no regions. The callee is a verified register
function from the existing function sequence. Const actuals must already be
initialized and exactly match widths. They may alias each other. Output actuals
must be fresh, width-exact, mutually disjoint and disjoint from all borrows;
they return initialized. Calls with unused Const parameters still require
initialized actuals. All definitions, including unused callees, must validate.
Reject duplicate symbols, unknown/wrong-kind callees, cycles, malformed types,
SSA/effects, outputs and argument roles before lowering.

Const means computational-basis values survive; phase and entanglement may
change. It is not a classical constant or an unchanged reduced density matrix.
Const references may be arithmetic inputs, phase inputs or nested Const actuals.
They cannot be initialized, computed into, passed to another fresh Output, or
targeted by qreg.gate. Even a cancelling X/X or H/H pair is refused: this slice
uses a conservative structural proof rather than arbitrary unitary equivalence.
Fresh outputs retain existing preparation/mutation rules. Existing arithmetic
and phase segments exactly clean their internal work before returning.

## Composition, resource and target boundaries

Compiler verifies the acyclic function graph, substitutes verified references
and threads the existing linear effect into a flat register model. Reuse the
existing arithmetic wrapper, schedules, scratch reuse, selection, checked
Program, Runtime and named reader. No state is cloned, evaluated or certified
externally. Qualified existing identities include the original callees; target
admission inspects reachable expanded phase actions, so a nested call cannot
bypass exact-phase guards and an unused phase callee does not restrict a root.

Cap helper call nesting at 32, source/model nodes at the existing 4096 bound,
and expansion visits at 4096 including calls that emit no effects. This prevents
an exponential tree of empty Const calls bypassing the emitted-operation budget.
Check final expanded nodes as well. Preserve source locations and the existing
fresh root profile. No locals, free/drop, consuming Input, native const, general
mutable roles, within/apply or arbitrary external unitary claims are admitted.

## Acceptance

Prove mixed Const/Output signatures, alias rules, unused borrows, invalid unused
definitions, recursion/depth/expansion limits, SSA/return obligations and basis
preservation. Execute all sixteen two-bit input pairs, complete complex states
with external correlations, repeated aliases, nested/repeated calls, caller
mutation after return, exact global signs and zero workspace under both
schedules. Compare direct and call producers behind the same execution consumer;
check exact phase target refusal, budgets and memory before execution. Include a
runnable existing public workflow and label new owned-IR examples contributor
only. Native source admission follows separately. Original corpus remains 3/3,
trials 0/2 and release_authorized=false.
