# Hamming-weight phasing source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `hamming_weight_phasing.ipynb` | `3279a78efaa2bd0bbf50eefe43d634612b830d7f` | Q-0667 |
| `hamming_weight_phasing.py` | `2c55d84b5b97f9eedbdcf78b9e4ea13d079161b1` | Q-0668 |
| `hamming_weight_phasing_test.py` | `bad2b4c5b0475e6b3232f2ebc6a3da34cdf21e6f` | Q-0669 |

## Original tutorial fixtures

Open [the native notebook](../../examples/notebooks/hamming_weight_phasing.ipynb)
with `jupyter notebook examples/notebooks/hamming_weight_phasing.ipynb`.
It reproduces both four-bit fixtures, types, gradient precision, and
top-level call identities. Both compile via `circuit.append(op)`.
The source exponent is `π/2` and the physical angle for weight `w` is
`π × exponent × w`. The original code comments about the phases of
`1111` and `0101` do not follow that exponent convention.

The gradient word is a through port. Its caller prepares a
negative-exponent Fourier gradient state to obtain phase kickback.
The compiler owns only the explicit count, junk, and addition carry
layouts. The notebook reports the bodies without the caller's gradient
preparation or unpreparation.

## Original numerical and resource grids

The direct grid covers all 28 source width/exponent combinations.
It compares the measured source stream and angles, full complex outputs,
and work return. For widths up to four, it projects every raw history.
For larger widths, two sparse executions check the complete supplied
complex state including work and junk. The five original large direct
counts and the original 5,000-bit T-complexity comparison are included.

The gradient suite covers all ten original cases and the default
high-precision fixture, in both directions. The source comparison
walks each original QVR and scaled-adder child, lowers the original
Add or its adjoint, and keeps the source offsets and feedback. The
arithmetic certificate checks all raw histories on every data word
and the tested gradient basis values, including wraparound. All
gradient values are used when the gradient has at most four bits.
Larger words use four representative gradient values. Actual sparse
executions check complete complex output vectors on those inputs.
The modular increments give the source Fourier phase and meet the
original epsilon for every possible data weight.

The locked original notebook passed. The bounded original rerun had
22 passes, eight source performance skips and four expected simulator
failures, with the ten gradient simulations deselected. The original
combined suite exceeded 240 seconds during those simulations. The
native suite covers their parameter grid using the independent source
leaf and arithmetic comparisons above.

| Four-bit fixture | Literal T/T† | Literal CX | Literal measurements | Source counts | Width |
| --- | ---: | ---: | ---: | --- | --- |
| Direct | 12 | 50 | 3 | 3 And, 35 Cliffords, 3 rotations, 3 measurements | 10 |
| Gradient | 1108 | 2592 | 277 | 561 Toffoli, 3 And, 35 Cliffords, 3 measurements | 79 reserved, 45 declared logical |

The gradient width is 35 bits. The compiler resource model composes the
original count and count adjoint with the original bounded number of
scaled-add calls. Its 45 logical wires exclude measured addition carries.
The native reserved width includes those carries. The source's
`phase_bitsize - 2` approximation also produces a negative Toffoli
declaration for a one-bit gradient. That count is preserved in the
source model and the literal gate inventory is reported separately.

## Remaining source-file boundaries

Q-0667 and Q-0669 are accepted. Q-0668 remains open for source
one-bit empty-junk uncomputation, mixed QFxp adjoint binding, and
recursive width estimation on the affected cases. The source
defects remain executable reproducers in the earlier focused suites.
Q-0666 remains open while its other rotation exports are audited.
