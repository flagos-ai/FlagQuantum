# Modular addition source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_addition.ipynb` | `b081513d2c657fb069e0da81f2b4914df531ba2f` | Q-0569 |
| `mod_addition.py` | `0a167e84fe00d5df50488c4b28925ea6114728f7` | Q-0570 |
| `mod_addition_test.py` | `7786a0ce6e07bf5ee7f7de410a0a41536e22ff3e` | Q-0571 |

The [native tutorial](../../examples/notebooks/modular_addition.ipynb) retains
all five operations, all ten source signatures and the five diagram and
call-graph sections. Its 27 Python cells run in an independent kernel.
The catalog uses the existing modular, controlled, scaled and constant
addition compiler functions through `circuit.append(op)`. Source call
declarations preserve child widths, activation values, constant parameters
and the allocated symbolic constant for scaled addition.

## Bounded constant addition

The locked `ModAddK` specifies the permutation on computational states,
including identity for out-of-range encodings. Its source resource graph
declares five equal-width Add calls. Qualification of a synthesized gate
stream uses the value map because the upstream operation supplies no
physical decomposition.

The native bounded emitter computes `x < modulus` into a clean membership
bit, combines it with the optional control pattern, and uses fixed-width
controlled constant addition. On an active residue, the wrap bit records
`x >= modulus - k` for `k` reduced modulo the modulus. Conditional
subtraction of the modulus produces the final residue. The equivalent
predicate `result < k` erases the wrap bit. The output preserves
membership, so repeating the original comparison clears its flag.
Classical slots are reused after each measured operation has restored
its work. A full binary ring uses the existing constant adder directly.

The tests certify coherent restoration for every basis input and all
measurement histories in eight small configurations, including mixed
controls, identity cases, negative constants and the inverse value map.
The frozen 64-bit example also checks fitting residues, the wrap point,
the first padding word and the maximal word without allocating a dense
state. A two-bit native append fixture for each of the five operations
checks complex amplitudes and zero-work output on three feedback histories.
Existing concentrated suites retain the gate order and feedback validation
for the other four source arithmetic chains.

| Native example | Logical ports | Clean work | Reusable slots | Emitted CX | Emitted CCX | Emitted T/T† | Emitted measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ModAddK(4, 7, 1)` | 4 | 9 | 4 | 276 | 2 | 120 | 30 |
| `ModAddK(64, 500, 23)` | 64 | 129 | 64 | 4734 | 2 | 2040 | 510 |
| `CModAddK(4, 7, 1)` | 5 | 14 | 8 | 239 | 0 | 84 | 21 |
| `CtrlScaleModAdd(4, 7, 1)` | 9 | 15 | 8 | 980 | 0 | 352 | 88 |

## Remaining source-file boundaries

Q-0569 is accepted. Q-0570 retains the locked `ModAdd` target-at-modulus
cleanup counterexample, the `CModAddK` out-of-range constant disagreement,
and the absence of a source primitive decomposition for `ModAddK`.
The new bounded permutation is an exact realization of the latter's
documented classical map.

The locked suite passed 446 cases including the notebook and all 356 slow
cases, with 18 skips and four expected failures. The skips include symbolic
decomposition and strict register-type boundaries. Native numerical callbacks
match all input tuples of the three slow grids. The original controlled
constant tensor blocks are matched via the certified native output map.
Q-0571 is accepted. Q-0568 exports other modular modules and remains open
until those dependencies are audited.
