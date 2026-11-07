# Classical controlled-bit callbacks

`simulation/classical_controlled_bits.py` ports the unmodified Qualtran v0.7.0
callbacks from `basic_gates/cnot.py`, `basic_gates/toffoli.py` and
`mcmt/multi_target_cnot.py` at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.
Compiler retains primitive emission and the existing fanout tree; Simulation
owns numerical evaluation. There are no external SDK production dependencies.

The three original policies deliberately remain distinct:

| Callback | Original rule | Diagnostic behavior |
| --- | --- | --- |
| CNOT | `(control + target) % 2` | NumPy promotion/overflow retained; no bit assertions |
| Toffoli | Assert target in `[0, 1]`, then compare each control with `1` | Left-to-right short circuit; inactive target identity preserved |
| Fanout | Test control truthiness, then XOR `2**width - 1` | Inactive width is never evaluated; nonzero control need not equal one |

The callbacks do not normalize operands or flatten arrays. In particular,
control value two flips the raw fanout but not CNOT's zero target or Toffoli's
target when it occupies the first control position. These diagnostic inputs
are not declared valid qubit states.

Run `python -m examples.algorithms.classical_controlled_bits`. It checks full
complex amplitudes for all valid-bit inputs entangled with a spectator and
restoration by the exact inverse through `circuit.append(op)`. No workspace or
measurements are required. The composed IR contains four CX and one atomic CCX,
with zero explicit T instructions. The source modern model instead classifies
CNOT and fanout as two logical Clifford components plus one Toffoli, or four T
equivalents. It does not certify a Clifford+T synthesis of the atomic CCX.

Reference tests compare types, broadcasting, original exceptions, integer
overflow and inactive object identity. Existing pinned-source gate-stream and
resource-category suites remain in the concentrated regression. Missing source
decompositions and raw callback diagnostics are not repaired or counted as
whole-file acceptance.
