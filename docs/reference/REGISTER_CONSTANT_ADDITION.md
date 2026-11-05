# Unsigned integer constant addition

`compiler/registers/arithmetic.py:add_constant_ir` ports the concrete unsigned
`AddK` call chain from Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Google copyright and Apache-2.0
source notices are retained. It loads `XorK` into clean quantum wires, applies
the shared `add_ir` implementation of `Always(Add)`, then unloads `XorK`.
Zero constants retain the complete source addition network; no independent
constant-specific optimization changes the source logic.

Inputs use big-endian bit order. For an n-bit target, callers supply n clean
constant wires, n-1 clean carry wires and n-1 distinct classical measurement slots.
All quantum work starts and ends in zero. The output is `(x+k) mod 2**n`.
Negative integers follow the source unsigned modulo normalization; nonnegative
constants must fit the target. Quantum wires must be disjoint nonnegative
integers. Controlled, signed, Montgomery and symbolic interfaces are not accepted
as completed by this slice.

Run without either reference package:

```sh
python -m examples.algorithms.register_constant_addition
```

The example uses `circuit.append(ConstantAddition(-3))`. All 16 sampled
trajectories map three-bit `|2>` to `|7>`, preserving phase and returning all five
quantum work wires to zero.

## Executable gates and independent model

The comparison gate set is X, H, S, CNOT, T/TDG, conditional CZ, measurement and
reset. Tests compare the unmodified source's gate multiset, each wire's complete
ordered dependencies, operand order, output map and measurement/feedback slots.
Upstream DAG export may reorder disjoint gates; no cancellation or rewriting is
used in the comparison.

| n, k | Measured T+TDG | Model And count | Model T at 4 T/And | Measured / model measurements | Constant + carry work wires |
|---|---:|---:|---:|---:|---:|
| 1, 1 | 0 | 0 | 0 | 0 / 0 | 1 + 0 |
| 3, -3 | 8 | 2 | 8 | 2 / 2 | 3 + 2 |
| 5, 0 | 16 | 4 | 16 | 4 / 4 | 5 + 4 |
| 8, 83 | 28 | 7 | 28 | 7 / 7 | 8 + 7 |

These model values come from `QECGatesCost`, with its explicit 4-T And conversion
via `total_t_count()`. The raw model `t` field is zero because it retains And
nodes; it is not an executable T count. The measured three-bit example emits
4 X, 6 H, 4 T, 4 TDG, 21 CNOT, 2 S, 2 conditional CZ, 2 measurements and 2 resets.
The eight-bit example emits 81 CNOT and 28 T/TDG in addition to the other gates.
Every emitted gate is also compared directly with executable upstream source.

The raw source Clifford model is **not** a full gate-level count. It omits
Cliffords inside And and uses a width-dependent Add expression that gives a
negative value at width one: `AddK(QUInt(1),1)` reports `clifford=-1`, while the
executable source and native IR both have two X plus one CNOT. This model
limitation is recorded rather than repaired or reported as equality. T and
measurement model agreement and complete executable gate equality are separate
acceptance claims.

## Local verification

42 new cases cover source gates and resources at widths 1, 2, 3, 5 and 8 with
zero, positive, maximum and negative constants. Native dynamic append tests at
widths 1 through 4 check every complex input amplitude and every carry outcome
observed in 64 trajectories, including the full clean workspace. Invalid widths,
constants, wire aliases and classical slots are rejected. All 313 combined
addition, polynomial, field, classical and native-phase regression cases pass.
The shared arithmetic module has 97% statement coverage; the new constant-add
function is fully covered. Strict mypy passes on 926 source files. Final whole-PR
remote checks and review remain independent of this local module acceptance.
