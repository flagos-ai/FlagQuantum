# Double-factorization PREPARE and SELECT file audit

Reference files are under `qualtran/bloqs/chemistry/df/` at Qualtran v0.7.0,
commit `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Source | Git blob |
| --- | --- | --- |
| Q-0343 | `prepare.py` | `49ea17ef2cc16ef8160ac80d2bc582d8437c0874` |
| Q-0344 | `prepare_test.py` | `c38337f74c47b0b5794210fca07a28be3f81b1ce` |
| Q-0345 | `select_bloq.py` | `9630186cd071c815e734ca71b3c6fa90bc887f1b` |
| Q-0346 | `select_bloq_test.py` | `2b6df8c79619b30f9ecfd191bd8eb2b3e9fedde1` |

## Declarations and ownership

The PREPARE model owns source signatures in their declared order. Inner
PREPARE exposes `xi, offset, rot, succ_p, p`. Outer PREPARE exposes
`l, succ_l`. Indexed data exposes `l, l_ne_zero, xi, rot_data, offset`.
Each port is through and the original convenience builder drops zero-width
words. The programmed-rotation model owns its explicit `offset, p,
rotations, spin, sys` signature. It retains empty opaque words and the
two-element system array in spin order. The models derive total declared
width from these signatures. They preserve forward and adjoint lookup
declarations and reuse the shared chemistry QROAM model.

The original PREPARE examples are 50 factors, with inner precision 7
and outer and indexed-data precision 18. The original OpenFermion
comparison at 108 spin orbitals, 360 factors and 13,031 eigenvalues
is covered in both call directions where those dimensions are relevant.
The model tests check child call order and counts, gate classes, live width,
and peak logical qubits against the pinned declarations. The unchanged
source suite reported 10 passed and nine skipped. All skips are
decomposition, graph-consistency, or dtype modes that require a source
decomposition.

## Executable evidence and resource accounting

FlagQuantum's algorithms own the table values and register roles. Compiler
components own lookup, arithmetic, swaps, and measurement-assisted erasure.
The executable components still use `circuit.append(op)`. Address words
are unchanged by lookup. Live payload words belong to the algorithm, and
work qubits return to zero after their last use. The resource declarations
above are independent of physical synthesis.

- Outer PREPARE and indexed-data tests compare the forward and reverse
  lookup primitive streams with the locked QROAMClean and swap components,
  and check complex amplitudes and erasure branches. The indexed-data
  five-row cases emit 44 and 12 T/T†, one and three measurements, and one and
  two work qubits for the tested blocked and unblocked configurations.
- Inner PREPARE checks the factor-local signed alias schedule, metadata
  lookup, and the adjoint. Its two-factor fixture emits 54 T/T†, ten
  measurements and six controlled RY rotations with three work qubits per
  direction. Small measurement histories are exhausted. The larger blocked
  case covers four representative histories.
- The executable SELECT test covers both spins, success and failure
  controls, the projected complex amplitudes and scratch cleanup. Its
  blocked three-spatial-orbital fixture emits 218 T/T† and 28 measurements
  with three work qubits.

The original four declarations omit composite implementations. There is no
source top-level primitive stream for those declarations. Q-0344 and Q-0346
are closed for the original runnable test modes.

The three native PREPARE examples retain the frozen display names
`In-Prep`, `OuterPrep` and `In_l-data_l`, the complete signatures and
the original cost models. The outer example's source rotation
precision is 18. Its declaration and cost depend on the alias precision.
Valid zero-bit metadata follows the source port omission rules.
The verified zero-precision PREPARE models still use the analytical
lookup domain. That lookup can reject undersized forward tables.

The programmed rotation declaration keeps zero-width words and
preserves the signed local Toffoli term. At the original 10-spin scale,
zero precision has local term -13 and source aggregate estimation
divides by zero. One bit has local term -3 and the source aggregate is
64 after adding the lookup contribution. The raw declaration preserves
the signed term and the lookup dimensions. The ordinary native
estimator admits nonnegative local counts and nonempty lookups,
including one bit for suitable dimensions. The prior executable cases
retain their own precision domain.

Before the complete declaration closure below, Q-0343 and Q-0345 retained
external-object and raw-declaration gaps outside ordinary estimator admission. The executable small-table
tests use the shared compiler lookup rather than the analytical
placeholder.

## Complete source declaration closure

Q-0343 and Q-0345 cover 18 definitions across the two complete source files.
Their actual external declarations now pass through existing Ecosystem/Compiler
library interchange and explicit native model consumers. No source algorithm was
edited. The original four classes declare signatures and analytical call counts
without executable bodies. Complete acceptance retains that boundary and reuses
the independently executable coefficient-driven components above.

| Complete source definitions | Native owner and evidence |
| --- | --- |
| `InnerPrepareDoubleFactorization`, `signature`, `build_call_graph`, `__str__` | `double_factorization_prepare_model`, `double_factorization_inner_prepare`, exact source display and original example tests. |
| `OuterPrepareDoubleFactorization`, `signature`, `build_call_graph`, `__str__` | `double_factorization_prepare_model`, `double_factorization_outer_prepare`, exact source display and original example tests. |
| `OutputIndexedData`, `signature`, `build_call_graph`, `__str__` | `double_factorization_prepare_model`, `double_factorization_indexed_data`, exact source display and original example tests. |
| `_prep_inner`, `_prep_outer`, `_indexed_data` | Existing unchanged-source example comparisons and `examples/algorithms/double_factorization_resources.py`. |
| `ProgRotGateArray`, `signature`, `build_call_graph` | `double_factorization_rotation_model` and the existing rotation/SELECT execution owners. |

`test_double_factorization_foreign_models.py` captures actual original constructor
libraries at depths zero, one and two in both directions for all four classes.
Owned constructor arguments feed the existing native signature and cost queries.
Original resolver reconstruction, registers, counts and all protobuf fields are
preserved. Protobuf count maps have no specified entry ordering, so the comparison
uses equality of parsed original messages and separately checks deterministic
owned-byte roundtrips. It does not claim byte identity with nondeterministic
original map serialization. The native estimator runs with Qualtran, Cirq and
Kickmix imports blocked after capture.

Actual signed inner-PREPARE and programmed-rotation declarations retain their
negative local Toffoli counts through the owned library and original resolver.
The nonnegative native estimator continues to reject these declarations at its
existing guard. Raw source declarations and an executable circuit have distinct
admission requirements. Small-table forward lookup and zero-precision source
failures remain as documented above. Negative widths, odd spin-orbital counts
and nonintegral dimensions remain outside the existing native model domain.
These boundaries are explicit dispositions, not pending physical implementations
of the original placeholders.

### Separate declaration models and executed fixtures

For 50 auxiliaries, 10 spin orbitals, 250 eigenvalues, 8 rotation bits and
12 alias bits, the unchanged-source and native models agree as follows.

| Declaration | Forward Toffoli | Adjoint Toffoli | And / Clifford / measurement | Rotation | Model peak |
| --- | --- | --- | --- | --- | --- |
| Inner PREPARE | 161 | 78 | 7 / 46 / 7 | 0 | 25 |
| Outer PREPARE | 62 | 33 | 17 / 111 / 17 | 2 | 24 |
| Indexed data | 46 | 15 | 0 / 0 / 0 | 0 | 26 |
| Programmed rotation | 235 | 99 | 0 / 0 / 0 | 0 | 62 |

Actual primitive counts for these original objects are unavailable because their
source classes have no executable decomposition. Both direct decomposition and
actual graph import preserve that error. The coefficient-driven fixture counts
in the earlier section remain independently measured evidence with different
input tables. Their full complex-amplitude, measurement-history, clean-work,
forward/inverse and normal append regressions are rerun as part of this batch.

The external receipt prefix is `g2-df-foreign-closure-`. It records all five
frozen source blobs, the complete definition lists, eight separate directional
resource rows, focused and concentrated tests, integration, strict typing,
formatting, architecture, IR, scope and API checks. The fresh original suite
reports 10 passed and nine source-owned skips for absent decompositions and the
associated graph/dtype checks. The original references retain their bytes.

The closure batch passed 218 concentrated cases, 17 local integration cases and
strict typing for the new foreign-consumer test file. Black, Ruff, Algorithms
scope, architecture, multi-level IR and protected API checks passed. The earlier
91-case chemistry/Hubbard repair regression remains separate accepted evidence.
Cirq 1.6.1 additionally passed all 78 declaration and foreign-consumer cases.
