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
are closed for the original runnable test modes. Q-0343 and Q-0345 keep the
remaining source metadata and admission differences open, including the
Python example/label surface and precision values outside the documented
native model domain. The original analytical QROAM itself can reject
undersized forward tables. The executable small-table tests use the shared
compiler lookup rather than the analytical placeholder.
