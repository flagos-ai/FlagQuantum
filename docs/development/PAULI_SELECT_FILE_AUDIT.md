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

Q-0591 and Q-0593 are accepted. Q-0592 now binds concrete foreign
dense-Pauli rows and serializes the resulting native logical and primitive IR.
Its source nonunitary coefficient domain and unsupported symbolic declarations
remain explicitly outside the admitted unitary execution path. The locked source applies
the sign of each coefficient's real part. Nonzero real parts of either
sign map to signed unitary rows, regardless of coefficient magnitude
or imaginary part. This includes infinite real parts and discarded nonfinite
imaginary metadata. NaN real parts, zero and pure-imaginary coefficients produce a
nonunitary controlled leaf in the source. The native circuit accepts
explicit signed Pauli rows.

The bounded path preserves the source's padded-address folding. The
ordinary native logical SELECT continues to apply identity on unused
addresses. Existing differential tests preserve the padded address
`11` with a three-row table and its exact complex phase.

## Complete external row binding

The complete locked implementation, original tests, example factory and direct
unary traversal were read. The Ecosystem-owned `select_pauli_lcu_from_cirq`
adapter consumes the external fields and converts every `DensePauliString`
immediately to a native signed Pauli word. Algorithms retains its existing
`SelectPauliLCU`; Compiler retains bounded traversal and resource ownership.
Cirq is imported lazily in the adapter and no external row is stored in native
IR. The adapter preserves the source real-part sign rule, with nonzero
real parts of either sign, including infinite magnitude and discarded nonfinite
imaginary parts. Zero-real-part and NaN-real-part rows are refused as nonunitary.
Invalid widths, target lengths,
control values, insufficient address bits, empty tables, foreign row types and
NaN real coefficients are refused before construction.

The existing four-row factory, Ising grids, source dependency comparisons,
operand and feedback checks, resource comparisons and all small tutorial
measurement histories now consume the actual adapter. Additional signed
I/X/Y/Z rows with complex coefficients test every operator column and every
measurement branch for no control and both control polarities. Full complex
amplitudes include the raw branch normalization and work bits must return to
zero. Logical IR and primitive IR each survive JSON serialization. These cases
reuse the existing append entry and the original self-adjoint source semantics.
The ordinary logical unused-address identity and source bounded padded-address
folding remain separate, already verified behaviors.

The resource table above is unchanged because the adapter changes input
representation, leaving the traversal and emitter intact. Its emitted primitive
inventory and independent locked And/Clifford model remain separate evidence.
Symbolic widths require explicit concrete binding. The original nonunitary
zero/pure-imaginary coefficient leaves remain a recorded source boundary, not
an accepted native unitary operation.

## Complete coefficient normalization dependency

Both Ecosystem conversion paths now use `_pauli_select_rows.py` for the exact
sign-of-real conversion. Actual untouched nth_operation leaves prove that an
infinite nonzero real coefficient still becomes a signed unitary Pauli, and
nonfinite imaginary metadata is discarded. The old public adapter rejected
all nonfinite metadata while the stored-Program capture rejected infinite real
parts only. Neither restriction followed the complete original conversion.
NaN real parts and zero real parts remain nonunitary source leaves and retain
their native refusals. No external metadata reaches the resulting signed words.

Twelve finite-coefficient operations and twenty-four complete directional
Programs remain exactly equal to published b6da4d703. Newly admitted inputs
produce the identical complete Programs of the corresponding normalized rows,
so unchanged traversal, call, feedback and work evidence is reused. A four-row
mixed bank additionally compares actual source primitive dependencies, ordinary
append, independent resource categories, complete complex operators and every
small raw measurement history in both directions and all three control modes.

| Mixed IX / -XY / YZ / -ZI bank | Actual T-family / CX / measurement instructions | Independent source And / Clifford / measurements / logical peak |
|---|---|---|
| No external control, either direction | 8 / 16 / 2 | 2 / 14 / 2 / 5 |
| Active on zero, either direction | 12 / 23 / 3 | 3 / 16 / 3 / 7 |
| Active on one, either direction | 12 / 23 / 3 | 3 / 14 / 3 / 7 |

This closes the actual coefficient-admission discrepancy without adding a
primitive or changing a protected Core contract. Source nonunitary leaves,
symbolic constructor failures and object-array serialization remain separate;
this dependency closure does not by itself accept the whole-file parent.

Final affected regression passes 63 cases in each Cirq 1.7/1.6 profile without
skips, plus 145 local integration cases. Strict typing covers five Python files;
format, lint, owner scopes, language/links, architecture, IR and API checks pass.
Optional-test imports are guarded; frozen source and actual Kickmix availability
are asserted on the reference lane. Dependency policy retains 724 repository
findings, none on this batch.

The complete constructor-domain intake additionally identifies distinct work:
Boolean width/control metadata and an empty target execute in the original
root graph and generator but are not fully admitted by both native paths.
Concrete SymPy integers execute in the source generator while its stored graph
classifies them as parameterized. NumPy integer widths and floating widths fail
in the original QAny validator. Empty selection fails in the original traversal.
These protocol distinctions are recorded in
g2-pauli-constructor-domain-intake-20261126.json in the reference evidence
directory. They are not repaired by silently casting every numeric input, and
the source's out-of-domain control value two does not become a new native
binary-control contract.
