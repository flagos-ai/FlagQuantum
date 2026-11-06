# Square cost phasing

The two native entry points bind the original
`Square → QVR → Square†` port graph through the shared cost
phasing composer. The result word stores the unsigned square.
The chosen fixed-point format interprets that word for the phase
stage. Both the result and clean arithmetic workspace return to
zero. The gradient route preserves the caller's negative-exponent
gradient state and permits sequential reuse of classical slots.

For the two-bit source, tests bind the original stage graph in
integer and four-fraction-bit formats. Direct rotations match
the original ZPow leaves and full source tensor, including its
precision truncation. The gradient route uses the original
composition's classical shifts and the validated scaled-add gate
binding. It certifies all measurement histories over each input
and gradient basis state. Sparse primitive trajectories also
check complex kickback and workspace cleanup with a coherent
spectator. Direct phasing projects every raw measurement history
and runs through `circuit.append(op)` in the local trajectory
executor.

The original Square leaf has no gate decomposition. Evaluation
and erase use the separately validated native column compressor
on the appropriate input subspaces.

| Cost format / QVR | Native T-family / CX / phases / measurements / wires | Pinned Toffoli / rotations |
| --- | --- | --- |
| Integer / direct | 16 / 34 / 4 / 4 / 8 | 4 / 4 |
| Integer / gradient | 24 / 55 / 0 / 6 / 13 | 5 / 0 |
| Four fraction bits / direct | 16 / 34 / 3 / 4 / 8 | 4 / 3 |
| Four fraction bits / gradient | 36 / 84 / 0 / 9 / 19 | 8 / 0 |

Coefficient and tolerance are both 0.125. Native counts exclude the
spectator and gradient preparation and unpreparation. Direct phase
counts are before synthesis. The source column is the original
abstract gate model.
