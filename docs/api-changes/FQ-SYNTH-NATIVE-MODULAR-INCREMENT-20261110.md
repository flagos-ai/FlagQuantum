# Native bounded modular increment contract

Status: implemented with local CPU evidence. Complete quantum-program and
Notebook coverage are tracked separately in the corpus catalog and delivery ledger.

## User semantics

For a declared, initialized raw unsigned `QNum` of width `w`, `q += k`
means `q <- (q + k) mod 2**w`, and `q -= k` means the corresponding
subtraction. The signed classical offset must be statically bound within
`-256..256`. This is an in-place permutation, with no fresh result allocation,
inferred widening or truncation of an ordinary arithmetic assignment.

The same statement can be placed in a static all-inout helper. A controlled
call to a helper containing exactly one such statement applies that permutation
only in the selected coherent branch. This does not admit arbitrary controlled
mutable helpers. Root and formal names may differ. The target must cover one
whole raw numeric register in logical bit order, disjoint from every condition
input and from other borrowed helper arguments.

## Compiler implementation boundary

Capture reuses verified `qreg.gate` X and indexed `qreg.control_x` effects.
The deterministic lower-bit ripple sequence for each unit increment runs from
high target bits to low; decrement reverses that order. At no point does the
native frontend import an Ecosystem source adapter. Predicate control reuses
existing bounded predicate compute/control/uncompute/release, including its
eight-input-bit limit and checked closed-scratch inversion. Raw scalar controls
retain their existing meaning. Static offset reduction is modulo the complete
target width; actual effect expansion and gate budgets remain authoritative.

Readonly, uninitialized, overlapping, partial or permuted targets, arrays,
signed/fractional formats, quantum offsets, dynamic loops and escaping ownership
changes refuse. Allocation and drop are never hidden inside a controlled call.
General helper inverse/control remains outside this contract.

## Acceptance

- Exhaust every basis value for bounded widths and offsets including zero,
  negative offsets, overflow and underflow; retain nonzero target values.
- Compare complete complex amplitudes against an independently constructed
  classical permutation Circuit, including inactive and active control branches.
- Verify coherent control and absolute global phases, bit order, helper name
  resolution, alias/readonly/uninitialized/format refusals and actual resource
  limits. Compiler scratch must be zero after each accepted operation.
- Preserve the original two-dimensional walk's two five-bit position words,
  two-bit coin and signed amplitude vector. Compare original/native/Circuit
  states at several step counts and nonzero inputs without phase alignment.
- Audit QLBM separately for nested conditions, field aliases, loop count and
  its original normalization refusal. Only its explicitly approved normalized
  revision may supply a normalized state; retain the unchanged original bytes
  and refusal. Do not credit this program until all these dependencies pass.
- Run a fresh full216 source replay and preserve preceding resource baselines
  before recording any additional native case acceptance. A primitive test is
  not original-program coverage or a complete Notebook workflow.
