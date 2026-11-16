# Pauli SELECT source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `select_pauli_lcu.ipynb` | `6513d5faf7b227cd26534f30e05a7d3b55a29f5a` | Q-0591 |
| `select_pauli_lcu.py` | `41e2f8ddb341fbbdd2d004ef533041bd24a5898c` | Q-0592 |
| `select_pauli_lcu_test.py` | `81ccdd543c60783cecee6a87ae238b050ed3b28b` | Q-0593 |

The [native tutorial](../../examples/notebooks/select_pauli_lcu.ipynb)
retains the four-row example, typed register signature, and all eight
direct call identities. It appends the bounded SELECT in uncontrolled,
zero-controlled, and one-controlled forms. Wires are ordered control,
big-endian selection, target in Pauli-letter order, then caller-owned
clean work. Measurement slots are explicit and the operation is
self-adjoint on the executable signed Pauli rows.

## Original grids and resources

The four-row tutorial suite checks the source gate dependencies,
operands and feedback in all three control configurations. A coherent
input spans every address, control state and target basis state. All
four uncontrolled and all eight controlled measurement histories are
projected directly, with full complex amplitudes and clean work checked.

The original four-site Ising cases cover all eight rows and both control
polarities, including inactive branches, using three raw histories for
each polarity. The three-site check compares the full output vector and
all eight original eigenstate overlaps against the source Hamiltonian
and preparation coefficients. The frozen source tests and notebook
yielded eleven passes plus the expected example serialization failure.

| Example | Emitted T/T† | Emitted CX | Emitted measurements | Source And / Cliffords | Source peak wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncontrolled | 8 | 17 | 2 | 2 / 18 | 7 |
| Active on zero | 12 | 24 | 3 | 3 / 20 | 9 |
| Active on one | 12 | 24 | 3 | 3 / 18 | 9 |

The declared model corresponds to 8 or 12 T equivalents at four T per
And. The source Clifford inventory is recorded separately from the
emitted CX and other physical instructions.

## Remaining source-file boundaries

Q-0591 and Q-0593 are accepted. Q-0592 retains the arbitrary foreign
dense-Pauli object and serialization boundaries. The locked source applies
the sign of each coefficient's real part. Nonzero real parts of either
sign map to signed unitary rows, regardless of coefficient magnitude
or imaginary part. Zero and pure-imaginary coefficients produce a
nonunitary controlled leaf in the source. The native circuit accepts
explicit signed Pauli rows.

The bounded path preserves the source's padded-address folding. The
ordinary native logical SELECT continues to apply identity on unused
addresses. Existing differential tests preserve the padded address
`11` with a three-row table and its exact complex phase.
