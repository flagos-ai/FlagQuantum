# Modular subtraction and package source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_arithmetic/__init__.py` | `774aa44d6c868ebc8eaa08386ac9b318b0d898de` | Q-0568 |
| `mod_subtraction.ipynb` | `e925323025d5e221d32d61c6974680b95c759d07` | Q-0578 |
| `mod_subtraction.py` | `e0d0fbf81a81ec2ed55e02b4b31ad3e75ab85324` | Q-0579 |
| `mod_subtraction_test.py` | `3d013fdd1c43b5853681c2920fd7296ba00bdedc` | Q-0580 |

The [native tutorial](../../examples/notebooks/modular_subtraction.ipynb)
covers the four source examples, their signatures, all direct children
and counts, and all eight diagrams. It also binds four concrete operations
and both polarities of the controlled ones via `circuit.append(op)`.
The operation binds control, big-endian residue inputs, and caller-owned
clean work in signature order. Only the negation target or subtraction
right operand changes. The adjoint uses the original inverse components.

The locked source suite passed 428 cases including the independently run
notebook. Two negation fixtures skipped strict type checks, and their
fixture count cases were expected failures. The native audit explicitly
compares these fixture signatures and declared children, including
zero-controlled X, fanout, and the controlled conjunction's release.
The prime residue grids run with unsigned and Montgomery source types
and both control polarities. The large seeded grid checks widths 32
and 33 at both source primes. Existing forward and adjoint suites cover
primitive operands, feedback, independent resources, complex amplitudes,
and work cleanup. The append audit composes every valid basis row through
the feedback certificate, and checks complete complex output and
adjoint restoration at width two. The existing small negation suites
also enumerate every measurement outcome.

## Resources

These are forward primitive counts for width three and modulus five.
Source CCZ equivalents are symbolic declarations evaluated at width three.

| Operation | Control activation | Emitted T/T† | Emitted CX | Emitted measure | Clean work | Source CCZ equivalents |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ModNeg | unconditional | 24 | 56 | 6 | 6 | 6 |
| CModNeg | 0 or 1 | 28 | 62 | 7 | 7 | 7 |
| ModSub | unconditional | 60 | 168 | 15 | 8 | 15 |
| CModSub | 0 or 1 | 80 | 221 | 20 | 9 | 20 |

## Package exports

The pinned source initializer provides the following 13 exports.
The audit checks their original import identities and their respective
FlagQuantum implementation owners under `compiler/registers/modular`.
Q-0568 is accepted at that ownership boundary.

| Source exports | FlagQuantum owner modules in the same order |
| --- | --- |
| `CModAdd`, `CModAddK`, `CtrlScaleModAdd`, `ModAdd`, `ModAddK` | `controlled_modular_addition`, `modular_constant_addition`, `modular_scaled_addition`, `modular_addition`, `bounded_modular_addition` |
| `KaliskiModInverse` | `modular_inverse` |
| `CModMulK`, `DirtyOutOfPlaceMontgomeryModMul`, `ModDbl` | `modular_multiplication`, `windowed_mod_mul`, `modular_doubling` |
| `CModNeg`, `CModSub`, `ModNeg`, `ModSub` | `controlled_modular_negation`, `controlled_modular_subtraction`, `modular_negation`, `modular_subtraction` |

## Remaining source-file boundaries

Q-0578 and Q-0580 are accepted. Q-0579 remains open. All-ones moduli
fail the source constant load because `p + 1` exceeds the word width.
The native emitter reports that boundary explicitly. At width two and
modulus two, the source callback preserves the exterior input two.
The source negation circuit instead produces zero with a dirty work
flag, and its release fails. The active branches of controlled negation
retain the same witness. The clean execution contract is the residue domain.
