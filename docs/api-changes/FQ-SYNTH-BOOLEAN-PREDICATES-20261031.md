# One-bit Boolean predicate composition

Date: 2026-10-31. The human instructed merging #975 and continuing the plan.
#975 merged as 731172aa795f486fe5a330c93fa84b60276deade after its exact head
2e26f4a9723403e4a9b603db514faaa4d2062a85 passed 57 checks with three configured
skips. Stage 2 of the sixth delivery plan includes Boolean expressions after
comparison and predicate-phase construction. This slice admits that expression
composition; it does not open quantum blocks or a new API. No independent trial
or GitHub approval is invented.

## User computation and domain

Python `and`, `or`, and `not` compose one-bit expressions: comparison flags,
UInt(1) inputs, one-bit helper returns, and classical numeric 0/1 constants.
Multi-bit integers are rejected rather than coerced to truth values, even in a
branch that ordinary Python would short-circuit. Boolean parameters and
True/False literals remain outside the unsigned declaration profile. No bitwise
operators, chained comparisons, new comparison kind, classical if/loop,
measurement, side effects or native Qmod grammar is admitted here.

For the admitted 0/1 operand domain, ordinary Python operand-valued and/or and
Boolean-valued not agree with the circuit's one-bit truth function. Capture
validates every operand in source order. Pure source expressions are built
coherently; compilation does not execute Python bodies or take a classical
short-circuit branch. A constant operand cannot hide unsupported source or an
unregistered call. The complete expression still has existing AST, operation,
width and helper-expansion ceilings. Source locations identify invalid operands.

## Reuse

Desugar AND to the existing unsigned product, NOT to equality with zero, and OR
to NOT(NOT a AND NOT b). Constant-only expressions fold exactly using their 0/1
values. Quantum inputs are never folded from their known preparation. Reuse
existing HybridProgram/SSA, qint.multiply/qint.equal, profile/ranges/lifetimes,
binding, retain/recompute, selection and exact prefix-adjoint phase cleanup.
No new opcode, kernel, type, public signature/export, report field, executor,
wrapper, schema or API snapshot. Services, Runtime and native-source dispatch
remain unchanged. Existing value/phase target boundaries remain unchanged.

Both value-mode XOR and phase-mode marked-sign effects retain all input values
and restore workspace. Repeated/aliased flags are reads of one encoding, not
cloned states. Full complex-state and entangled-reference tests compare absolute
amplitudes without dividing out a global phase. Resource reports cover the
selected whole circuit, including composed predicate temporaries and cleanup.

## Acceptance

- Both schedules: complete Boolean truth tables, nested and multi-operand forms,
  aliases, helper/constant composition, and numeric flag reuse.
- Real CPU value XOR for both destination bits, including coherent target/input
  states and an external reference; phase absolute sign, complex preparation,
  interference and clean workspace through the public Program/run/read path.
- All 16 pairs of two-bit inputs for a composite comparison, public value and
  phase, both objectives; actual costs and plan identity checked.
- Multi-bit truthiness, unsupported syntax/calls hidden by constants, recursion,
  source/expanded budgets and resource constraints fail before execution.
- Existing source, single-comparison, arithmetic, phase and original Qmod
  acceptance remain valid. Public API/proposal baselines remain byte-identical.

Independent human trials remain 0/2 and release authorization remains false.
