# Rz via phase gradient source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `rz_via_phase_gradient.ipynb` | `75e37a757fb74d78b06704638dd34a78cb42c7c2` | Q-0684 |
| `rz_via_phase_gradient.py` | `d3ec380d24af7dfb8a115b6fe88074ad53e22e93` | Q-0685 |
| `rz_via_phase_gradient_test.py` | `4fb5e79871ad3943caad86206771575039146f7d` | Q-0686 |

Run `jupyter notebook examples/notebooks/rz_via_phase_gradient.ipynb`.
The native notebook preserves the original ten cells and appends the
four-bit operation through `circuit.append(op)`. It renders the
signature and source-declared call graph and prints both the literal
gate inventory and the symbolic resource formula. Both notebooks ran in
independent kernels, and the native kernel rendered two SVGs with
Qualtran, Cirq and quimb imports forbidden.

## Source binding and tensor evidence

The qubit, angle and phase gradient are through registers. The bits of
both fixed-point words are cast through unsigned integers at the same
width. The default branch subtracts on zero and adds on one. The adjoint
flips those roles. The source graph performs two casts on entry,
one ControlledAddOrSubtract call, and two casts on return. The tests
compare the original operations, physical operands, and feedback slots
at the same arithmetic lowering depth.

For a raw angle word `a` and a negative-exponent Fourier gradient of
width `b`, the body applies `Rz(4πa/2**b)` to the data qubit.
The caller supplies the angle and Fourier gradient. Carries begin at
zero and return to zero in every measurement branch. The original
three- and four-bit tensor cases prepare angle words 1 and 2. The
native tests compare two independent complex input vectors to the full
source tensor and restore all initialized registers on every raw
history. The four-bit body also covers every angle word, gradient basis
word, and control value in both directions.

The unchanged source suite returned eight passes and one expected
failure from the source counts autotester. The original source symbolic
cost assertions were checked separately against the shared Add model.

| Four-bit body | Literal T/T† | Literal CX | Literal measurements | Source QEC categories | Reserved / source wires |
| --- | ---: | ---: | ---: | --- | --- |
| Forward | 12 | 41 | 3 | 3 And, 28 Clifford, 3 measurements | 12 / 12 |
| Adjoint | 12 | 41 | 3 | 3 And, 26 Clifford, 3 measurements | 12 / 12 |

For a symbolic equal width `n`, the source forward model uses `n-1` And,
`9n-8` Clifford, and `n-1` measurements. The adjoint removes the two
polarity flips. The symbolic live width follows the source maximum over
the input, the Add child and each controlled complement. One-bit
Clifford counts preserve the original abstract Add anomaly.

## Remaining source boundaries

Q-0684 and Q-0686 are covered. Q-0685 remains open for generic dtype
admission at the native binding and symbolic execution. The source Add
validator raises for incomparable symbolic widths, such as angle width
one with an unconstrained symbolic gradient width. The resource query
preserves that admission boundary. Q-0666 remains open for the other
rotation exports.
