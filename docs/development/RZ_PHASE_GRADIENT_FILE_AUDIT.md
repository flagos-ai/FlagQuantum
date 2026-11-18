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

Q-0684 and Q-0686 are covered. Generic dtype bit binding and explicit
resolution of symbolic widths are now exercised below. The source Add
validator raises for incomparable symbolic widths, such as angle width
one with an unconstrained symbolic gradient width. The resource query
preserves that admission boundary. Q-0666's export inventory is accepted.

## Whole-file generic object binding

The entire source class, factory, documentation specification and original test
file have been read. The two dtype fields feed the source signature and its two
unsigned-cast properties. Native `RegisterCast` preserves those same raw words;
`rz_via_phase_gradient_ir` delegates the arithmetic to the existing measured
controlled-add/subtract owner. The source body has four casts and one arithmetic
child, while its declared call graph omits the casts. The native example and
independent symbolic model preserve that distinction. The factory, original
tensor helper, symbolic cost assertion and notebook map to the previously
accepted tutorial and file tests.

`test_rz_source_binding.py` creates the original class with symbolic width fields,
rejects unresolved physical execution, and explicitly resolves its dtypes. Actual
`QUInt`, `QAny`, mixed fractional `QFxp`, signed `QInt`/`QFxp`, and bounded
`BQUInt` objects bind in both directions at widths (1,1), (2,2) and (2,3).
The original casts determine the unsigned bit interpretation. Signed numerical
values are not reinterpreted as signed arithmetic. Bounded padding columns
exercise the physical circuit extension; they do not enlarge the bounded
register's declared value domain. A one-bit signed `QFxp` is invalid in the
source constructor, so that case uses the valid one-bit `QInt` type.

Imported and native emitters use the same `BlackBoxPrepare` append consumer.
Owned JSON IR, every logical matrix column, all-history channel identities,
independent final-zero checks, exact opcode histograms and true adjoint recovery
agree. Every raw angle and both data values also satisfy the analytic complex
phase when applied to the negative-exponent Fourier gradient. Each direction's
independent resource model agrees with the unchanged source. Existing operand,
dependency and measurement-feedback evidence is reused at the accepted lowering
depth. Angle width greater than gradient width remains rejected by the original
arithmetic admission rule.

| Actual source widths (angle, gradient) | Forward / adjoint instructions | T/T† | CX | Measurements | Physical wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| (1,1) | 5 / 3 | 0 | 3 | 0 | 3 |
| (2,2) | 26 / 24 | 4 | 13 | 1 | 6 |
| (2,3) | 50 / 48 | 8 | 26 | 2 | 8 |

The one-bit abstract Add Clifford anomaly is retained in the source model;
the physical table is measured separately. Q-0685 therefore retains this
inherited model boundary and the incomparable symbolic-width admission case.
The new generic and resolved-symbol binding evidence does not claim execution
of an unresolved symbolic circuit or close the parent while those boundaries
remain in its ledger.

The final regression passes 656 cases in 39.25 seconds. Strict typing, Black,
Ruff, Algorithms scope and diff checks pass. This tests/audit-only batch reuses
accepted architecture, IR, protected API and independent kernel evidence.
An initial fixture tried the source-invalid signed one-bit `QFxp`; replacing
that fixture with the valid `QInt(1)` preserves the source admission rule.
