# Register subtraction with measured carries

`compiler/registers/subtraction.py` ports the executable `Subtract` and
`SubtractFrom` call chains from Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Google copyright and Apache-2.0
source notices are retained. Both functions compose the existing `add_ir`;
no second carry or measurement-cleanup implementation is introduced.

The directions are deliberately distinct:

- `subtract_ir(left, right, prefix, carries, classical_bits, signed=False)`
  preserves a and writes **a-b** into b. It follows the source identity
  `a-b = ~(~a+b)`, including the input complement and restoration order.
- `subtract_from_ir(left, right, carries, classical_bits)` preserves a and
  writes **b-a** into b. This equal-width operation follows the separate
  source chain `~b; Add(a,b); ~b`.

All quantum registers are big-endian. For `subtract_ir`, left width m must
satisfy `1 <= m <= n`, where n is right width. Callers supply n-m initially
zero prefix wires and n-1 clean carry wires, plus n-1 distinct nonnegative
classical measurement slots. With `signed=True`, the input is sign-extended
using the literal tree-layer decomposition of `MultiTargetCNOT`; otherwise it
is zero-extended. The prefix and carry wires are all restored to zero.
`subtract_from_ir` requires equal widths and uses only carry workspace.

The output wraps modulo `2**n`. Signed right interpretation does not change the
bit pattern. No new dtype or runtime allocator is introduced. Only concrete
uncontrolled interfaces are accepted by this slice; symbolic and general
controlled construction remain pending.

## Runnable append example

```sh
python -m examples.algorithms.register_subtraction
```

The example needs neither reference package. `circuit.append(op)` runs signed
two-bit a=-1 and three-bit b=2 through a-b, producing the three-bit pattern 5
(signed -3) with all three work wires zero. Repeating a-b restores b=2.
A separate equal-width b-=a example computes `2-3 mod 8 = 7` and clears both
carry wires.

## Executable counts and source model

| Variant | Measured CNOT | Measured X | Measured T/TDG | Model T at 4 T/And | Measured / model measurements | Clean prefix + carries |
|---|---:|---:|---:|---:|---:|---:|
| a-b, 3 to 3 bits | 21 | 9 | 8 | 8 | 2 / 2 | 0 + 2 |
| a-b, unsigned 3 to 8 bits | 81 | 24 | 28 | 28 | 7 / 7 | 5 + 7 |
| a-b, signed 3 to 8 bits | 99 | 24 | 28 | 28 | 7 / 7 | 5 + 7 |
| b-=a, 8 bits | 81 | 16 | 28 | 28 | 7 / 7 | 0 + 7 |

Each complete executable source stream matches the native gate multiset and
each wire's ordered dependencies, including operands, measurements and feedback.
The gate set includes X/H/S, CNOT, T/TDG, conditional CZ, measurement and reset;
no gate cancellation, sign-extension simplification or SWAP insertion is used.
Disjoint gates may commute in the upstream DAG exporter.

`QECGatesCost` retains And nodes. T model values use its explicit
`total_t_count()` conversion, with 4 T per And; measurements are compared
separately. Its raw Clifford count does not represent the fully decomposed
network. For unsigned/signed 3-to-8 subtraction it reports 70/72 Clifford
units, whereas literal signed extension adds 18 executable CNOTs. The source
model counts two multi-target Clifford operations at that coarser level; it
also retains And decomposition overhead separately. These raw values are not
asserted equal to primitive Clifford totals. Complete executable source and
native gate counts, unlike these mixed-level model fields, agree directly.
The earlier width-one Add model boundary limitation is not repaired here.

## Local acceptance

49 new cases cover executable source and independent resource comparisons for
QUInt, QInt and QMontgomeryUInt. a-b uses widths (1,1), (2,2), (3,3), (5,5),
(8,8), (1,3), (2,5), and (3,8); b-=a uses widths 1, 2, 3, 5 and 8.
Native small-register append cases compare all input-pair complex amplitudes,
including an entangled spectator, all sampled carry outcomes and complete work
cleanup. a-b is tested as an involution, and b-=a is reversed with the shared
adder. Validation rejects aliases, invalid prefix/carry layouts and classical
slots. No upstream source patches are used in these comparisons.

The combined regression passes 386 tests. The subtraction module reaches
100% statement coverage; strict mypy passes on 927 source files. Both supported
Black versions, Ruff, architecture boundaries and multi-level IR checks pass.
Whole-PR CI/GPU and review remain independent of this local module acceptance.
