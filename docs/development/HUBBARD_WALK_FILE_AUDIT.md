# Hubbard walk factory and tests

The frozen Qualtran v0.7.0 files are under
`qualtran/bloqs/chemistry/hubbard_model/qubitization` at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Source | Git blob |
| --- | --- | --- |
| Q-0355 | `walk_operator.py` | `ba3f1e4402af26c6317aeb96bfd8343a581649be` |
| Q-0356 | `walk_operator_test.py` | `84322cee42884cdbbecb479c0d6ccdee63bf3326` |

## Factory contract

`HubbardWalk.from_lattice(side, t, u)` constructs the original square-lattice
walk with the selection-only zero mark. It composes the shared native
PREPARE, SELECT and reflection owners through `circuit.append(op)`.
`source_layout` exposes the source selection and system word identities
in deterministic native binding order. The source's selection union uses
a set. Port comparisons therefore match by name and width. The source
allocates the two `temp` bits in the reflection. The native binding
provides those wires explicitly and tracks them separately from scratch
and feedback. `HubbardWalk(prepare)` retains its full-signal mark for
callers that include direction bits in the initial signal.

## Execution and accounting

- The original 2×2 protocol test passed. It checks PREPARE and SELECT
  decomposition validity, both SELECT control polarities and diagram
  labels. The original notebook test is covered by independent execution
  of the frozen Hubbard notebook in the preceding batch.
- The factory normalization, visible word widths, two-stage graph and
  source logical resources match at sides 2, 3 and 5. The 2×2 test binds
  the actual source reflection operands, including the prepared direction
  bits, zero-polarity controls, phase target and scalar −1. The comparison
  maps the source scalar to a phase on the initialized reflection target.
- The walk adjoint reverses SELECT and reflection. SELECT reverses its
  Majorana, exchange and spin-up stages and conjugates the scalar.
  Each clean measurement-assisted stage continues to compute predicates
  before erasing them. Native SELECT forward/adjoint cases include no
  control, positive control and negative control on sides 2 and 3.
- Four measurement histories verify a 2×2 narrow walk and adjoint on a
  complex superposition, checking the entire restored state and clean work.
  The existing full-signal fixture also verifies the projected SELECT,
  the second Chebyshev step and adjoint recovery.
- At 2×2 the narrow model has peak 26 logical qubits. Native emission
  gives 224 T/T†, 28 measurements and eight work qubits for the narrow
  walk, and 232 T/T†, 30 measurements and ten work qubits for the
  independent full-signal fixture.

Q-0355 and Q-0356 close the original square-lattice factory and tests.
The separate PREPARE and SELECT inventory entries track the side-one,
padding, larger literal gate comparisons and source Majorana count
boundaries. This batch compares the walk's source composition and
reflection operands and reuses their previously verified primitive paths.
