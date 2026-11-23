# Integer arithmetic constructor dependency audit

## Complete source scope and ownership

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Complete `arithmetic/addition.py`
and `arithmetic/negate.py`, their original tests, and the direct
`bookkeeping/always.py` dependency and tests were read. Seven frozen source,
test and resolver blobs are verified in
`ftqc-reference-env/g2-arithmetic-constructor-intake-20261123.json`.
Existing complete-file arithmetic, bitwise, control, output ownership and exact
adjoint evidence is reused. The constructor layer introduces no arithmetic
algorithm or numerical codec.

| Constructor | Native semantic owner | Ports and ownership |
|---|---|---|
| Add | `registers/arithmetic/addition_component.py` | Caller-owned a and b, MSB first, a width at most b width, clean carries |
| AddK | `registers/arithmetic/constant_addition_component.py` | Caller-owned x, source-encoded constant load, unconditional Add, exact unload |
| Negate | `registers/arithmetic/negation_component.py` | Caller-owned x, ordered BitwiseNot and AddK(1), clean constant and carries |
| OutOfPlaceAdder | `registers/arithmetic/out_of_place_addition_component.py` | Caller-owned a/b, produced c on RIGHT or consumed c on LEFT, explicit ownership transfer |
| Always | `registers/primitives/always_component.py` | One unconditional call with the exact child ports and direction |

Compiler `formats/legacy_arithmetic_factories.py` validates explicit constructor
metadata and reuses those existing component owners. Ecosystem binds the same
Simulation stored-value encoder used for bitwise constants. The common MSB
pattern validation is shared with the word constructor adapter. There is no
Compiler-to-Simulation import or duplicate integer encoding.

Always is constructed through its existing owner using the resolved child's
declaration view. Only the wrapper root is returned to the resolver. The resolver
already retains the real child and all helper definitions under their assigned
names. This supports nested wrappers and children with helper calls without
reflection, inferred names, copying foreign objects or inventing another closure
registry. The original wrapper's control bypass protocol remains covered by the
existing native Always component tests.

## Exact source boundaries

Add supports signed, unsigned and Montgomery descriptors, including mixed widths.
A narrower signed source operand is zero-extended by the original circuit.
Classical signed arithmetic and the physical stored-word circuit remain distinct.
Unsigned and Montgomery negative AddK constants are reduced modulo the stored
width before invoking the existing encoder, matching the original load operation.
Positive overflow and signed out-of-range constants retain source refusal.

Negate(QInt(1)) cannot encode its original AddK(1) dependency. The source raises
while serializing a decomposition. A depth-zero native constructor also validates
that original constant and refuses. The existing separately documented native
stored-pattern Negate extension is not presented as a successful original
constructor path.

Symbolic integer widths and constants, including Sympy integer literals, remain
logical declarations. They refuse physical binding through the existing owner.
Zero-width arithmetic retains existing native physical refusal. A serialized
CompositeBloq at depth zero omits required connections, so executable decomposed
libraries use depth one or greater. Named source constructors are tested at all
three retained depths.

Generic Adjoint/Controlled/Power constructors remain separate open resolver scope.
The existing native exact-adjoint and both-control-polarity evidence is reused,
including exhaustive small branch histories. Constructor acceptance does not
claim those generic wire wrappers are resolved. Source OutOfPlaceAdder.__pow__
drops the carry flag for power minus one; constructor direction tests use the
original .adjoint() method, which preserves it. The reference remains unchanged.

## Execution, phases and resources

Named and directly decomposed libraries are compared gate by gate, including
operands and measurement feedback, allowing only commuting independent gates.
Reversed words retain caller wire order. Full complex amplitudes with an
entangled spectator are compared with independent original tensor contractions.
All small raw measurement histories preserve their unnormalized complex phase
and return private work to zero. Complex64 and complex128 trajectories use the
normal circuit.append(op) path. Directional output isometries explicitly transfer
and consume their output ownership and restore the full initial state after
measured erasure.

For Add with right width n, actual primitive T/TDG and measurement/reset counts
are 4(n-1) and n-1. AddK and Negate retain those carry counts with width n and
2n-1 clean constant/carry work bits. The extra actual X gates follow encoded
loads and complement. OutOfPlaceAdder uses n-1+carry And outputs, with four T
per output forward and one measurement/reset per output in its adjoint. Its
returned sum is logical output, separate from private clean work.

Primitive scans and unchanged QEC models are checked separately. The original
one-bit Add model has its known Clifford-count defect. The source call graph also
uses equal-width Add Clifford counts for narrower inputs. Those model limitations
remain registered in existing native arithmetic audits. Passing primitive streams
does not assert those analytical estimates equal every executed inventory.

## Local acceptance evidence

The concentrated affected regression passes 1,184 cases. Local runtime
integration passes 145 cases. The coverage selection passes 217 cases, measuring
100 percent statements and branches across the arithmetic, bookkeeping and word
constructor adapters and Ecosystem constant binding. Strict typing covers the six
changed Python files. Black, Ruff, both team scopes, repository language,
architecture, IR and public API checks pass. The dependency checker retains
exactly the 720 previously recorded findings and exits one.

The unchanged original non-notebook tests pass 281 cases with ten original
protocol/symbolic skips and six original xfails. Both original notebooks pass
with explicit existing kernel dependency paths. Their initial sandbox socket
failures remain in the intake log. The notebook rerun uses local kernel port
permission and retains the original sources. Seven frozen blobs and the native
Kickmix 0.1 extension with callable Simulator are verified in
ftqc-reference-env/g2-arithmetic-runtime-20261123.json.

Receipts and complete logs are retained as g2-arithmetic-quality-20261123.json,
g2-arithmetic-regression-20261123.log, g2-arithmetic-coverage-20261123.log,
g2-arithmetic-original-nonnotebook-20261123.log and
g2-arithmetic-original-notebooks-20261123.log. All twelve original addition and
negation examples resolve at retained depths. Constant admission is compared
against original stored encoders over the complete small signed domain and
positive/negative overflow boundaries.

Five additional complete identifiers bring built-in constructor coverage to
37 of 303, with 266 remaining, including partial OnEach and scoped Adjoint.
Q-0876 remains open. Pending Proposals 116-119, generic wrappers, GPU, performance
and whole-repository qualification retain their existing dispositions.
