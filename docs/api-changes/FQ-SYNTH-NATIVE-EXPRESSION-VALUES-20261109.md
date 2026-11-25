# Native comparison and bitwise values

The user authorized native counterparts for every example in the five categories,
including missing operations and successive local batches. This narrow unreleased
language extension reuses existing comparison and bitwise IR, verification,
lifetimes and lowering. It changes no Stable Core export, execution contract or
serialized schema. Release authorization remains false.

Unsigned Python qfunc expressions may compute comparisons (`==`, `!=`, `<`, `<=`,
`>`, `>=`, including chains) into fresh output storage. The result is one bit.
Ordering annotation methods return QBit; equality keeps Python's existing bool
annotation, which is accepted as an int expression by the capture-only assignment
marker. The function body is never executed and quantum values are never host
Booleans. Existing coherent-control predicates use the same expression capture.

Unsigned `&`, `|` and `^` use the existing word operation: operands zero-extend;
AND has the smaller encoding width and OR/XOR the larger width. An integer
literal has its minimal positive unsigned width. `~x` complements exactly the
inferred quantum encoding width of x, including leading zeros; it is not Python's
unbounded negative integer complement. Classical-only binary operations fold
exactly; classical-only inversion is refused because it has no quantum width.
The shared pure-expression frontend has the same rules. Python logical
and/or/not retain their one-bit-only boundary. Signed/fractional values, shifts,
negative literals, arbitrary host calls and numeric array arithmetic remain
outside this profile. Fresh assignment retains its zero-storage ownership rule;
it does not become in-place XOR on already initialized outputs.

## Acceptance

- Migrate the six original unsigned comparator/bitwise examples without loading
  Qmod from user examples. Independently compare raw complex statevectors,
  logical bit order and zero workspace, including coherent inputs.
- Check all six comparisons on every pair of small unsigned inputs, width and
  zero-extension boundaries for bitwise values, both operand orders and fixed
  width inversion. Compare independent Circuit permutations and preparation.
- Reject initialized output reassignment, self-aliasing, uninitialized reads,
  out-of-budget circuits, negative constants and widthless inversion. Keep
  public API snapshots and all frozen original source/resource records intact.
- Record native program coverage separately from complete Notebook workflows.
