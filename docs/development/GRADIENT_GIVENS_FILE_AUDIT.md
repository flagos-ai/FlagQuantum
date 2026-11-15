# Phase-gradient Givens rotation audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/quad_fermion`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0394 | `__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` |
| Q-0395 | `givens_bloq.ipynb` | `c0f4c81e2eeb77a7179c132bfb45b5ef7e817729` |
| Q-0396 | `givens_bloq.py` | `8546f518da85a1e36f24ed95caf19428cf368e6c` |
| Q-0397 | `givens_bloq_test.py` | `1ae077366155d297f6bb1c070adc3bb0ca1bdd1f` |

The source package initializer contains a license header and exports no
objects. The native locations are the compiler phase-gradient folder and
the algorithm primitive for binding. Its operation attaches through
`circuit.append(op)`. The existing programmed rotation used by THC
and double factorization retains its original angle convention and circuit.

## Binding and execution

The real form has two target bits and two same-width unsigned fractional
words. The complex form uses separate real and complex angle words.
All words are big endian, and all logical ports are through. At width
`n`, the caller supplies `n-1` clean carries and matching classical
slots. Each Rz returns those carries to zero. The next Rz can reuse
the same slots.

The compiler emits the source 12-gate Clifford ladder around the two Rz
calls bound to the same real angle word. The complex form adds a third
Rz following the ladder and binds only its complex angle word to the
second target. The adjoint reverses the unitary Clifford subsequences and
uses the directional add/subtract leaf for the Rz operations. The
reference comparisons traverse original register connections, casts,
adder gates and carry feedback. They check physical operands and
per-wire order for real and complex, forward and adjoint, at widths
1, 2, 4 and 8. The widest fixtures use permuted wires.

At width two, the ideal state is computed by replacing each source Rz
with its analytical two-dimensional unitary on the negative-exponent
gradient. All real and complex angle values enter coherently. The tests
compare the full unnormalized branch vector with no phase adjustment,
including expected zero amplitudes on nonzero carries. They enumerate
four real and eight complex histories in each direction.

## Counts and educational assets

| Eight-bit case | Emitted T and T adjoints | Emitted measurements / resets | Source modeled And / measurements | Source modeled Cliffords | Source peak logical qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real forward | 56 | 14 / 14 | 14 / 14 | 140 | 25 |
| Real adjoint | 56 | 14 / 14 | 14 / 14 | 136 | 25 |
| Complex forward | 84 | 21 / 21 | 21 / 21 | 204 | 33 |
| Complex adjoint | 84 | 21 / 21 | 21 / 21 | 198 | 33 |

The model instantiates the shared source Add declaration. It retains
the source one-bit Clifford-count anomaly and its control-flip budget
difference between forward and adjoint. Emitted primitives are counted
on the native instruction stream separately.

The unmodified original tests passed all nine non-notebook cases with
OpenFermion 1.6.1. The unchanged 18-cell notebook ran independently.
The native notebook reproduces both four-bit signatures and call graphs,
and includes an append example with resource counts. Its separate kernel
checks four SVG outputs without importing upstream quantum packages.

The concentrated regression passed 283 cases across phase-gradient,
programmed Givens, THC, and double-factorization consumers. Strict
typing, format, ownership, architecture, IR contract, and public API
checks passed.
