# Complete phase estimation of quantum walks notebook audit

The frozen `phase_estimation/phase_estimation_of_quantum_walk.ipynb` has blob
`6937510fa2d03e83463fbd9c1443c77827c3a981` at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. All fifteen original cells map
one-to-one to `examples/notebooks/quantum_walk_phase_estimation.ipynb`, plus
one evidence-boundary Markdown cell. Algorithms owns the example helper
`examples/algorithms/quantum_walk_phase_estimation.py`. It reuses admitted
walk, PREPARE, SELECT, QPE, QFT, register binding, cost and drawing owners.

## Complete cell mapping

| Source cells | Native responsibilities |
| --- | --- |
| 0–4 | Original physical motivation, ideal sine tensor and notebook-specific execution schedule |
| 5 | Six-site Ising, epsilon 0.01, four phase bits, full ordered stage table and call graph |
| 6–7 | Source circuit helper's QEC categories, explicitly excluding ideal initialization and QFT |
| 8–9 | Complete 200-site Ising resource query, epsilon 1e-5 and fourteen phase bits |
| 10–11 | Complete 20-by-20 Hubbard query, t=20, u=80, 800 spin orbitals and computed width 21 |
| 12 | Separate full included QPE model, including LP preparation and QFT |
| 13–14 | Small Hubbard full-QPE historical budget, folded T paths and actual FlameGraph SVG |

The source display is represented by the owned ordered stage table and model
call graph. The final renderer accepts `FLAMEGRAPH_EXECUTABLE`, an explicit
local FlameGraph executable. Acceptance uses the locked local executable.
Folded repeated paths accumulate in a Counter before rendering. This retains
source T mass while avoiding materializing hundreds of thousands of repeated
path strings. The complete original notebook runs unchanged under IPython,
including its time magics, in a bounded reference subprocess.

## Distinct execution ordering and phase acceptance

The notebook controls the first phase wire directly, then the second, third,
and subsequent wires control zero-reflection pairs around powers 2,4,8,... .
The shared QPE class schedules from the final phase wire. The helper reverses
only the schedule's binding view and restores the original word order for QFT.
It uses the existing QPE composition and independent directional callbacks.
Users append the result with `circuit.append(phase_estimation(...))`.

Both the source and native acceptance inject the identical ideal sine tensor
as initial input. The native helper explicitly binds identity PREPARE at that
boundary. Physical synthesis of the ideal initialization channel is excluded.
The zero-bit tensor utility remains the valid singleton [1]. Circuit binding
requires a positive width.

Widths 1,2,3 compare every applicable source stage and operand, complete source
Power exponents, and full complex output tensors on coherent walk inputs with
a complex spectator. Each direction has one feedback measurement. All two
forward and four independent forward/inverse histories pass, with scratch zero
and absolute phase preserved. The reference contracts complete original stages,
including Power objects. It unwraps SDK adapters only when they lack their own
tensor contraction, preserving the actual power. It never phase-aligns states.
The accepted child and QFT gate-stream evidence is reused in batch regression.

Actual captured-child primitive totals are 35,105,215 at widths 1,2,3. Each
uses one clean work qubit and one feedback slot. The two-bit inventory is
12 identity, 28 X, 5 H, 2 T, 20 CX, 2 T-dagger, 6 S, 4 CZ, 1 measurement,
1 reset, 15 opaque matrix leaves, 3 Z, 4 phase, 1 swap and 1 controlled phase.
These inventories include explicit identity preparation and source capture
matrix leaves. They belong to these small execution fixtures.

## Independent complete model queries

| Original query | CSwap | And | Clifford | Rotation | Measurement | T |
| --- | --- | --- | --- | --- | --- | --- |
| Six-site Ising circuit | 168 | 1474 | 11825 | 84 | 1474 | 0 |
| 200-site Ising circuit | 295362 | 21321353 | 124158198 | 65636 | 21321353 | 0 |
| 20-by-20 Hubbard circuit | 134219344 | 4391438157 | 7270849440 | 33555056 | 4391438157 | 0 |
| Included full Hubbard QPE | 134219344 | 4391438389 | 7270850016 | 33555359 | 4391438389 | 19 |

All categories match untouched original queries. The native composition reuses
existing uncontrolled and positive-controlled SELECT/walk models, independent
PREPARE costs, open-control reflection costs, LP categories and textbook QFT
costs. The full-QPE query includes costs intentionally omitted by the circuit
helper. This notebook asserts gate categories only, with no large-state peak
workspace or primitive-lowering claim.

The small Hubbard full-QPE flame graph uses the source historical T budget,
with 7 T per CSwap. Its folded weights close at 3542140 T, and 131428 rotations
remain separate. The original's own direct historical Clifford count is
9167472, whereas its QEC-to-historical conversion is 8314766. Native QEC
conversion matches the latter exactly. Both untouched source paths agree on
T and rotations. The historical Clifford discrepancy remains evidence of the
two source accounting methods. It is not hidden by a correction to source or
by treating a converted budget as the original direct legacy result.

## Remaining boundaries

Original large Ising/Hubbard factories remain declarations and resource queries.
Large physical QPE admission and SDK-default leaf synthesis remain under Q-0631.
The direct historical Clifford decomposition path is a separate unclosed metric
from the accepted QEC conversion. The original paper Fig. 2 discrepancy and
Q-0634/Q-0622 gaps remain recorded. No large physical amplitude, clean-work or
primitive-count conclusion follows from these model-only cells.

The final concentrated regression passes 236 cases, including the full QPE,
walk, reflection, window and QFT acceptance. Strict mypy covers the complete
example helper, extracted notebook and test. Black, Ruff, Algorithms scope,
architecture, multi-level IR and protected API snapshot checks pass. The owned
schedule graph and final flame SVG were rendered and visually inspected.
External evidence uses the `g2-quantum-walk-` prefix.
