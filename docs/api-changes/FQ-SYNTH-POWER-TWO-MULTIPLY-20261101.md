# Exact multiplication by positive powers of two

Follow #999 at 57883bcda1aa6fe54b3e3989f65c5de44c58c9a0. Optimize only
existing qint.multiply with one quantum unsigned operand and one classical
constant 2**k for k>=1, when the operand's complete inferred upper bound is
positive. Use its existing result allocation and copy the variable into the
high result bits, leaving the low k bits zero. No new language, opcode, API,
IR, arithmetic kernel, allocator, dependency or executor. No direct output
fusion or normalization of classical powers is added here.

Skip generic multiplication scratch only for that proven case. Read current
operand bindings for forward, inverse and rematerialized computation. Preserve
exact inverse, copy-out to arbitrary XOR targets, cleanup, schedules, resources
and memory guards. Both operand orders work. Zero, one, non-power constants,
two quantum operands and derived zero-bound values keep generic multiplication.
Never narrow domains from preparation, samples or user assertions.

Acceptance: exhaustive small inputs and arbitrary nonzero XOR targets on
permuted bindings, both schedules, complex/reference-entangled states, shared
and rematerialized dependencies, nested inverse cleanup, Python/native producer
replacement, explicit fallback and budget boundaries and complete actual cost.
Execute x*256 under a bounded public 64 MiB budget and retain a below-requirement
memory refusal. Original source/license evidence, trials 0/2 and unreleased
status remain unchanged.
