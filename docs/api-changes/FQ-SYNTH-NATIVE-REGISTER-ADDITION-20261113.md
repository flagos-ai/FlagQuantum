# Native unsigned register modular updates

This compiler extension supports the ongoing authorized complete native example
migration. It adds captured Python behavior to existing numeric operators, with
no export, callable signature, serialization or Stable Core snapshot change.
The preceding constant-offset contract remains the historical record of that
slice. This document defines the additive register-source profile.

For initialized, physically disjoint raw unsigned words, `target += source`
means `(target + source) mod 2**len(target)`; subtraction uses the same modulus.
The source is preserved, including when borrowed as `Const[QNum]`. Widths may
differ. High source bits vanish modulo the target width; this does not imply
truncation of ordinary fresh arithmetic assignments. Static source slices and explicit
packed source views use logical low-bit-first ordering. Every selected view must
contain 1..256 distinct locations. Disjoint slices of the same physical owner
are accepted through checked inout helper arguments. Existing Const/writable
helper views still require separate owner registers. Direct indexed augmented
assignment retains its existing syntax refusal. Any overlap is refused even if a particular value would be safe.

Capture reuses the existing checked unit increment/decrement and indexed
controlled-X effects. Each source bit controls an update of the corresponding
target suffix. Existing Compiler lowering owns MCX work and its cleanup; this
frontend adds no carry-storage allocator, opcode, runtime kernel or dependency.
Effect expansion, operation, width, depth and memory budgets remain binding.
This deterministic implementation makes no resource-optimality claim.

Signed/fractional storage, uninitialized sources or targets, Const targets,
compound quantum addends, duplicate or out-of-range views and overlapping
physical storage refuse before execution. A plain static helper can borrow the
two registers; checked adjoint inverts its captured effects. Controlled helpers
retain their existing single-port constant-update boundary: this extension does
not admit two-register updates under a coherent helper call.

Acceptance exhausts small unequal/equal width basis pairs, including nonzero
targets and wraparound; compares independent complete complex permutation
Circuit states without phase alignment; checks source preservation, inverse
round trips, sliced/packed ordering, alias/format/initialization refusals and
resource ceilings. The tutorial preserves its two-bit uniform source and
three-bit accumulator initially four. Full Notebook acceptance separately
requires all original teaching stages, actual sampling and source mapping.
