# Retained-difference comparisons and explicit logical outputs

This batch ports the executable Qualtran v0.7.0 call chain at commit
`bcb17bb25360ed68ce3bc084645db3e929799389` in
`qualtran/bloqs/arithmetic/comparison.py`:
`_HalfLinearDepthGreaterThan -> _HalfComparisonBase -> LinearDepthHalf*`.
Google copyright and Apache-2.0 notices are retained. No upstream algorithm is
patched. Symbolic widths and generalized controls remain outside this slice.

## Ownership and intermediate contracts

`compiler/registers/half_comparison.py` owns `half_greater_ir`. It composes
shared sign extension, BitwiseNot, truncated OutOfPlaceAdder and Xor, without
copying their gate implementations. Equal positive-width operands are
big-endian, with concrete unsigned or two's-complement interpretation.

Forward takes zero `difference` (n+1 bits), zero `target` (one bit) and two
zero `signs` work wires. It computes `(b-a) mod 2**(n+1)` and `a>b`, restores
the physical operands, and clears both sign work wires. Difference and target
are populated logical outputs, not clean scratch. The measured adjoint requires
the correct complete pair; it clears both outputs, preserves physical operands
and clears sign work, using n caller-supplied classical slots. It is not an
arbitrary-dirty-output XOR comparator or blanket unitary inverse.

`compiler/registers/half_ordering.py` owns `half_compare_ir` and
`half_compare_outputs`. The four relations reuse the same half comparator:

| Relation | Base input order | Target flip | Retained difference | Executable adjoint logical output ports |
|---|---|---|---|---|
| gt | a,b | none | b-a | a=left, b=right |
| le | a,b | after compute / before erase | b-a | a=left, b=right |
| lt | b,a | none | a-b | a=right, b=left |
| ge | b,a | after compute / before erase | a-b | a=right, b=left |

All differences wrap to n+1 bits. `half_compare_outputs` records logical port
bindings for the same validated layout passed to the emitter. Physical operand
wire states are restored in every case. The ge/lt adjoint port exchange is
retained from the actual pinned source, not silently corrected to its classical
model. Tests explicitly compare forward and reverse output bindings as well as
gates. A minimum asymmetric example (a=1,b=0) proves that the source ge/lt
classical adjoint returns (1,0) while its decomposed logical ports return (0,1).
Both signed and unsigned examples are retained. That source model/decomposition
discrepancy remains open; a claim that both source contracts agree is forbidden.

## User example

```sh
python -m examples.algorithms.register_half_comparison
```

The user writes `circuit.append(HalfComparison(relation,signed))`. The two-bit
example computes a retained comparison, copies its target to a separate result,
then erases the temporary pair. It verifies all physical input/result complex
amplitudes, all four relations in both signed modes, and six temporary wires
returning to zero. It prints logical adjoint port bindings separately, including
the source ge/lt aliases. No reference package or NumPy is needed at runtime.

## Measured primitives and source estimates

Let s be 1 for signed extension and 0 otherwise; f is 1 for ge/le and 0 otherwise.

| Path | Measured CNOT | Measured X | Measured T/TDG | Source QEC T | Measurements / source model |
|---|---:|---:|---:|---:|---:|
| Forward | 11n+3+4s | 3(n+1)+f | 4n | 4n | 0 / 0 |
| Measured adjoint | 5n+3+4s | 3(n+1)+f | 0 | 0 | n / n |
| n=8 unsigned compute+erase | 134 | 54+2f | 32 | 32 | 8 / 8 |
| n=8 signed compute+erase | 142 | 54+2f | 32 | 32 | 8 / 8 |
| Two-bit unsigned example including result copy | 39 | 18+2f | 8 | 8 | 2 / 2 |
| Two-bit signed example including result copy | 47 | 18+2f | 8 | 8 | 2 / 2 |

The caller binds n+2 logical output wires and two clean sign work wires;
complete erasure clears all n+4 temporary wires. The retained result in the
example is separate. These are explicit register budgets, not claims of
allocator-minimized peak qubit usage. The source graph may reuse names from
nonoverlapping allocation lifetimes. Reference tests use dedicated output names
and verify the actual port map; this allocator policy does not patch gates or
justify a peak-memory equivalence claim. Primitive Clifford counts are measured
separately from source coarse call-graph estimates. The base source model also
omits explicit sign-extension overhead, while its relation wrappers account for
it; T/measurement agreement is checked independently for both source entry types.

## Verification scope

The base suite has 22 cases: 15 source width/dtype pairs (widths 1,2,3,5,8 and
unsigned, signed, Montgomery bit representations), six coherent small-width
cases, and invalid-input checks. The relation suite has 91 cases: 60 source
width/dtype/relation pairs, 24 coherent small-width cases, relation/binding
validation, four source-port counterexamples and the runnable example.

Forward and reverse streams are compared separately, including operands,
per-wire dependency order, metadata, feedback and RIGHT-output bindings. The
reference allocator preserves dedicated output names; least-significant-first
Bloq discovery is explicitly bound to big-endian result words. There are no
algorithm patches or altered source comparisons.

For widths 1–3, complete coherent operand amplitudes, signed extremes, equality,
correct difference direction and forward sign-work cleanup are verified. Every
measurement outcome is postselected through the actual primitive erasure IR
without normalization. Each branch equals the original complex input divided
by sqrt(2**n), checking phases, restored physical operands, zero outputs/work
and branch probabilities. The projection helper is shared with Hamming-weight
tests; no simulation or Runtime implementation was added to Compiler.

Batch acceptance: 113 new and all 960 combined regression cases pass, including
the pinned Kickmix native reference, with no skips. Both new modules have 100%
statement coverage. Strict mypy passes on 943 source files; minimum/current
Black, Ruff, architecture boundaries and multi-level IR contract checks pass
locally. Final integration CI, GPU and review remain separate gates.
