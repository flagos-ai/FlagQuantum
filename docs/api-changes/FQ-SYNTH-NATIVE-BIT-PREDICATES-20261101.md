# Native one-bit logical composition

Follow #996 at 8a6d976580b975b5c3550a49f210719bc453dc44. Admit &, |, ^
and unary ~ only for operands whose complete inferred unsigned domains fit one
bit, including comparison expressions and classical 0/1. Multibit operands are
refused even when initialized to small basis values. Literal-only expressions
fold; quantum-dependent constants retain their reads and initialization checks.
These are coherent eager operations, never host Boolean conversion, measurement
or short-circuit evaluation. Signed complement and in-place ^= remain refused.

Use precedence **, unary ~, *, +, &, ^, |, comparisons (highest first).
Comparison composition should normally use parentheses. Preserve the existing
expression/tree/token/expanded-model guards, including unary nesting depth.
Use existing private AST records and qint operations: AND is multiplication,
XOR is inverted equality, OR is positive-sum comparison, NOT is equality to zero.
Compiler alone owns kernel choice, cleanup, aliasing, lifetimes and resources.
No new public API, opcode, canonical IR, kernel, executor or dependency.

Acceptance covers all truth tables and aliases in value/phase modes under both
objectives, nested expressions, comparison flags, complex reference entanglement,
interference, constant folding, width refusal, eager uninitialized reads,
precedence/depth budgets and explicit-arithmetic producer replacement. Preserve
source/license digests, four-original cohort, trials 0/2 and unreleased status.
