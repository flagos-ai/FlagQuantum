# Coherent conditional expressions in unsigned synthesis

Date: 2026-10-31. The human instructed merging #979 and continuing the sixth
plan. Its head 95f357212b56c5dd184c925c914e78e8c67d2500 passed 57 checks
with three configured skips; merge is 94cfd28837b1a63b6938e0ecd2ab4fd682c7d2db.
This is the pure value-selection slice of stage 2, behind the existing Python
synth entry. Public signatures, schemas, contracts and snapshots do not change.

## Source semantics

`a if predicate else b` selects an unsigned value coherently for every basis
input. The predicate must be an unsigned one-bit value, a supported comparison/
Boolean expression, or a classical integer 0/1. No generic integer truthiness,
measurement or classical sampling is introduced. Both branches are checked as
pure supported expressions, even for constant predicates. Classical predicates
fold to the chosen expression; already captured quantum work in either branch
is retained and cleaned conservatively. This is declarative expression capture,
not execution of Python branches or short-circuit side effects.

Nested expressions, explicitly registered helpers, aliases and static loop
assignments compose under existing budgets and immutable-borrow rules. The
result width is the maximum of the branches' proven unsigned ranges, without
truncation. A one-bit selection remains one bit and can be used as a predicate
or by the existing phase mode. Branch values may have different widths or be
classical literals. No signed or fixed-point values, statement-level if/else,
mutable branch assignments, general control/within-apply or runtime feedback
are admitted. Source errors precede planning, including errors in unselected
branches.

## Internal representation and reuse

Add one private `qint.select(predicate, true_value, false_value)` operation in
the existing HybridProgram/SSA. Its predicate is quantum.uint(1); each branch
is an existing quantum.uint or nonnegative INDEX value. The verifier and
lifetime analysis use max(true_bound, false_bound), and track all quantum
operands as inverse dependencies. Existing binary add/product cannot express
exclusive alternatives without overestimating width and destroying the one-bit
phase contract; this operation is the minimal domain-specific representation.
No new type, region, identity, contract object, executor or serializer is added.

The current arithmetic emitter allocates only the selected result width, with
zero additional kernel scratch. It XORs b, predicate*b and predicate*a into
fresh result storage with X/CX/CCX, then uses the existing XOR-output and exact
cleanup. Branches zero-extend. Repeated predicate/source aliases simplify
identical controls; borrowed inputs are never flipped to create a negative
control. The actual emitted prefix is exactly inverted for phase mode.

Both existing retain/recompute schedules, finite candidate selection, final
target/resource checks, immutable Program, fq.run and named reading remain
authoritative. Core/Algorithms/Services/Runtime/Simulation and native Qmod
syntax are unchanged. Release authorization remains false and independent
usability trials remain 0/2.

## Acceptance

Compare independent Python selection references over complete small basis
inputs and every XOR target, including unequal widths, literals and repeated
aliases. Verify arbitrary complex amplitudes and an external entangled
reference, both schedules and full workspace restoration, without dropping
global phase. Public tests cover numeric min and one-bit phase predicates,
helpers/qfunc/loops, uniform/complex preparations, supported target conversion,
resource/memory rejection and no planning after source errors. Refuse forged
SSA arity, guard types, attributes, regions and insufficient result widths.
Record actual total/workspace/operation/depth costs. Preserve original-size
Qmod regression and existing comparison/Boolean/loop/phase contracts.
