# Bounded Python comparison synthesis

Date: 2026-10-31. The human user explicitly approved proceeding with the concrete
proposal to add == and < after merging #971. This record defines that narrow
unreleased-language extension before implementation. It does not authorize a
new public entry, export, result schema or source serialization. Existing API
snapshots and the historical Qmod proposal remain unchanged.

## Semantics and scope

A single Python == or < compares full unsigned operands by numeric significance,
with zero extension for unequal widths. Operands may be supported arithmetic,
immutable quantum input borrows, registered helpers or nonnegative constants.
Both-constant comparisons fold to exact 0/1; quantum operands are never folded
from a chosen preparation. Logical aliases and repeated operands retain their
identity and phases. Comparisons return one quantum.uint(1) value encoded 0/1.
The existing evaluator computes z ^= predicate(inputs) for either initial target
bit and zero workspace, preserving input/relative phases and cleaning workspace.
A comparison result can be reused as its one-bit numeric encoding by existing
arithmetic. There is no measurement or classical branch on a quantum value.

Optional bool return annotations declare a one-bit result, including in helpers;
non-one-bit returns are refused with source locations. Parameter annotations
remain optional int. Literal True/False, chained comparisons, !=, <=, >, >=,
Boolean and/or/not, Python if/loops and implicit conversion remain unsupported.
Qmod grammar/version admission does not change. No Bool facade/export is added.
Named probability results remain integer tuples containing 0 or 1.

## Reuse and ownership

Compiler reuses existing HybridProgram/SSA, binding checks, dependency/lifetime
analysis, retain/recompute scheduling, candidate selection, resource ceilings and
target lowering. The pure comparison gate kernel is extracted from the existing
Algorithms prefix-equality comparator into Compiler's shared arithmetic-kernel
boundary, alongside the existing adder. Algorithms keeps its public validation
and greater-than behavior while consuming the same kernel. Compiler never
imports Algorithms, Circuit, Runtime or a backend for gate construction.
Services keeps the same synthesize/Program/run/probabilities workflow. Source
acquisition, seven synth exports, public signatures, defaults and report fields
remain unchanged. No truth-table enumeration or evaluation of user Python bodies.

## Required acceptance

- Independent integer references for all 16 two-bit x/y inputs, including
  x*x+y < 5 and equality, preserving both operands and an arbitrary XOR target.
- Unequal widths, constants in either position, equal/aliased operands and
  boundary values; optional bool annotations and helper composition.
- Complete complex amplitudes for coherent phase-bearing input and nonzero
  target; both schedules and clean workspace at tolerance 1e-12.
- Actual resource estimates, logical and line target compilation, checked
  ceilings/repair failures and explicit construction budgets.
- Existing Algorithms comparator and consumers retain numerical behavior;
  invalid syntax, profile records and physical bindings fail before execution.
- Existing public Python/Qmod acceptance, API protection and architecture pass.

Two independent developer trials remain 0/2 and release authorization false.
This adds predicates to the existing bounded arithmetic profile. General
controlled blocks, invert/within-apply, effectful qfuncs and all-corpus Qmod
compatibility require later separate admission.
