# Proven carry-free positive constant addition

Follow #998 at 4d5771af4585144d5247a243890235ad4318c298. In the existing
arithmetic scheduler, admit a cheaper implementation of qint.add only when one
operand is a positive classical constant and its set bits lie wholly above the
other operand's complete inferred unsigned domain. Use the conservative mask
(1<<unsigned_width(upper_bound))-1. Test both operand orders. Never infer the
proof from basis initialization, sampling, gates, or a user assertion.

For a proven case allocate only the full result width and XOR-copy the variable
encoding and constant into it. Disjoint bits prove sum equals XOR. Preserve the
existing result/copy-out/cleanup schedule, exact inverse, source locations and
event accounting. Overlapping masks, zero constants and two quantum operands
retain existing ripple lowering. No new opcode, kernel, IR, API, scheduler,
dependency or executor. No root-only direct-output shortcut.

Acceptance: both schedules, all inputs and arbitrary nonzero XOR targets on
noncontiguous bindings, coherent phases/references, inverse cleanup and zero
work, shared/rematerialized dependencies, nested scopes, source-producer
replacement, fallback boundaries, complete actual resources and memory guards.
The existing x+2**8 memory refusal must use a budget below the new actual
requirement; add a positive public execution under a bounded 64 MiB limit.
Preserve source/license evidence, trials 0/2 and unreleased status.
