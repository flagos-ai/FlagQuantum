# Native unsigned comparison expressions

Date: 2026-11-01. Continue the sixth-plan conditional-oracle dependency after
#989 (57 successful checks, three configured skips; merge
1283555b482ca2e6ac0e727e66531fa00bf00615). Const/Output calls are now admitted;
this slice lets their bodies describe comparisons without hand-designed gates.
Consuming Input/local ownership remains a separate contract, not a shorthand
for erasure. Only Ecosystem production code changes.

## Expression semantics

Admit unsigned binary ==, !=, <, <=, > and >= expressions, with lower precedence
than +, * and literal **2. Compare complete unsigned domains with the existing
Compiler zero extension. Each operand is evaluated once in source order and
the result is integer 0/1 encoded in one quantum bit. Literal-only comparisons
fold to exact classical 0/1; quantum expressions retain their declared domains
independently of prepared values. Results can initialize fresh Output scalars
or compose with already admitted arithmetic.

Reject unparenthesized comparison chains, identity/membership, logical/bitwise
operators, Boolean literals/types and statement-level control. Parenthesized
binary comparisons can be nested explicitly; no Python chain semantics are
silently assigned to native source. No arbitrary Python or source evaluation.

Reuse qint.equal/qint.less, reversing operands and complementing a one-bit
result through existing qint.equal-with-zero for the other relations. Reuse
the existing verifier, calls, lifetimes, comparator kernels, exact cleanup,
schedules, selection, Program, run and named reader. There is no new Compiler
operation, public type, API signature/export, report schema or dependency.

Native phase keeps its polynomial source boundary: a comparison in a phase
argument is refused, including nested comparisons. Compute a fresh one-bit
flag first, then use phase(flag, pi). That flag is a live named Output, not an
implicitly cleaned temporary. Its output correlation and internal zero-work
guarantee are separate. Existing exact-phase target restrictions remain.
The public Python predicate output_mode path is unchanged.

## Source evidence and scope

Pinned upstream comparator_integer_example.qmod uses res = a <= 2, three-bit
uniform a and an inferred result width. Retain its bytes and Git blob/SHA-256
in an additive comparison-cohort sidecar, reusing the retained original MIT
license. Preserve the baseline three-file manifest, fixtures, verifier and API
contract byte for byte. Only actual full-size file/text execution may add this
fourth original to parsed/lowered/executed/full-state counts. The separately
inspected two-variable comparator original requires signed multi-argument
allocation and remains unsupported; do not rewrite it to claim compatibility.

## Acceptance

Exhaust both two-bit input domains for every relation, including literals on
either side, unequal widths, aliases and classical folding. Check source
precedence/chains and errors before planning; verify owned comparator records
inside Const helpers. Execute coherent flags, complex phases on exported flags,
external correlations and zero internal work under both schedules. Replace a
native producer with the existing Python/owned producer behind the same consumer.
Exercise text/Path, target/budget/memory checks, full-size unmodified new
original, existing originals and provenance corruption refusals. Record actual
costs and a runnable user example. Existing Input/Const role boundaries, scope
gaps, independent trials 0/2 and release_authorized=false remain explicit.
