# Explicit permutation cycles through shared bit patterns and equality

The pinned executable source is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`:
`XorK.controlled() -> PermutationCycle -> Permutation` in
`arithmetic/bitwise.py` and `arithmetic/permutation.py`.
Google copyright and Apache-2.0 notices are retained. Compiler emits native IR;
it does not allocate runtime resources or introduce reference-package types.

## Responsibilities and contracts

`compiler/registers/bitwise.py` adds
`controlled_xor_constant_ir(control,target,value)`: one positive coherent
control, a nonempty big-endian target and a fitting unsigned bit pattern.
Each selected source X becomes a CNOT in the same bit order; no hidden work,
measurement, T gate or control reduction is introduced. Zero constants emit no
gates after validation. Other control patterns are not this concrete entry.
The existing plain/controlled AddK load/unload paths now consume shared
`xor_constant_ir` / `controlled_xor_constant_ir`; their original streams are
unchanged, including outer control-polarity flips.

`compiler/registers/permutation.py` owns `permutation_cycle_ir` and
`permutation_ir`. For target width n, the caller supplies one initially zero
marker, n-1 initially zero carries, and n-1 distinct classical slots. Size N
satisfies 1 <= N <= 2**n. Each cycle is nonempty with distinct entries in [0,N).
For every cycle entry v, the original sequence is retained:

1. XOR target==v into marker using shared EqualsAConstant.
2. Apply XorK(v XOR next(v)) controlled by marker.
3. After the loop, perform the final equality against the first cycle entry.

Every equality clears its carries with the source measured And erasure. Marker
and carries are zero after each full cycle. Singleton cycles retain both source
equalities; no identity or constant-specific optimization is applied.

Explicit nonempty cycle sequences execute in source order. Cross-cycle overlap
is allowed and makes order observable. Marker/carries/slots are reused after
cleanup. Dense/partial-map conversion, symbolic cycles/widths, empty cycle
sequences and generalized controls remain outside this slice.

The physical circuit leaves all basis values outside a cycle unchanged, including
values >= N. Tests check that executable extension without claiming that such
values satisfy the source BQUInt classical input domain.

## Runnable user path

```sh
python -m examples.algorithms.register_permutation
```

The user writes `circuit.append(BasisPermutation(cycles))`. The example applies
ordered overlapping cycles ((0,2,5),(2,7)) to a three-bit target while preserving
an entangled spectator. It checks every complex target/spectator amplitude,
three work wires returning to zero, and inverse composition restoring the input.
It uses two reused classical slots for 14 measurement events. No reference
package or NumPy is required at runtime.

## Primitive measurements versus source model

For explicit cycles let E=sum(len(cycle)+1), and let D sum the popcounts of
successive XOR deltas, including the closing delta of each cycle.

| Path | Measured CNOT | Measured T/TDG | Independent source model T | Measurement events / source model | Clean quantum work | Reused classical slots |
|---|---:|---:|---:|---:|---:|---:|
| General explicit cycles | (6(n-1)+1)E+D | 4(n-1)E | 4(n-1)E | (n-1)E / (n-1)E | n | n-1 |
| 8-bit cycle (3,70,131,255) | 233 | 140 | 140 | 35 / 35 | 8 | 7 |
| 3-bit example ((0,2,5),(2,7)) | 101 | 56 | 56 | 14 / 14 | 3 | 2 |

The primitive gate set includes X/H/S, CNOT, T/TDG, measurement, conditional CZ
and reset. Source QEC T/measurement costs are compared independently of expanded
primitive scans; coarse Clifford or symbolic estimates are not substituted for
those scans. These are explicit register budgets, not allocator-minimized peak
claims. A reused classical slot contains its latest value, not the full event
transcript. The number of measurement events remains unchanged; the reference's
fresh keys are explicitly bound to those consumed/reused native slots.

## Gate-stream and phase evidence

The controlled-bit-pattern suite has 33 cases: five widths, four constants,
complete small coherent control/target/spectator permutations, involution and
invalid-input checks. The existing plain/controlled AddK consumers and original
bitwise suite are regressed together; 225 focused cases pass before batch
regression.

The cycle suite has 27 cases: nine concrete width/size/cycle settings compare
both a single cycle and source composition, seven complete small coherent
settings enumerate all branches, validation and the user example. Source
settings cover width 1/2/3/5/8, N=1, non-power-of-two N, singleton and nontrivial
cycles. Original Bloq decomposition is flattened before primitive expansion,
preserving the measured And call chain rather than choosing generic unitary
synthesis. Operands, per-wire gate dependencies, feedback and independent costs
match. Temporary allocation names are bound per equality lifetime: carry first,
marker at its first copy, then reused after cleanup. No source algorithm or gate
is patched; this name policy does not certify minimal peak allocation.

The seven small settings use all basis inputs with nonuniform complex phases
and an entangled spectator, including singleton and overlapping-cycle cases.
Every measurement event is independently enumerated in both forward and inverse
execution, even when its storage slot is reused. Unnormalized primitive-IR
postselection equals the ideal complex permutation divided by sqrt(2**events),
checking phases, full work cleanup and branch probabilities. Enumerating only
the final stored bits would not prove the earlier feedback branches and is not
used. Symbolic/dense-map conveniences and whole-file acceptance remain open.

Batch acceptance: 60 new and all 1,020 combined regression cases pass, including
the pinned Qualtran and Kickmix references, with no skips. Shared bitwise and
permutation modules have 100% statement coverage; strict mypy passes on 944
source files. Minimum/current Black, Ruff, architecture and multi-level IR checks
pass locally. Final integration CI, GPU and review remain separate gates.
