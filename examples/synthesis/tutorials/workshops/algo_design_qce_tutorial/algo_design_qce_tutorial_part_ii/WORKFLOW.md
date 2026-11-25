# Native Knapsack helper composition

Two named unsigned numbers represent item counts: `a` uses three bits and `b`
uses two. Preparation, constrained value phase and mixer helpers compose into
one native quantum program. All three original cost/mixer layers are retained.

```python
import flagquantum as fq
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.main import build_program

program = build_program((-0.17, 0.23, -0.31, 0.41, -0.53, 0.67))
result = fq.run(program)
print(dict(program.probabilities(result, registers=("a", "b"))))
print(program.report["resources"])
```

The first three parameters set the value phases, and the last three set RX
mixers. The shared cost helper applies `-3*a-5*b` only when `2*a+3*b <= 12`.
Mixers reuse Pauli evolution: commuting X terms at time beta/2 produce RX(beta)
on each logical bit. No approximate Trotter error occurs for these commuting
terms. Users do not select physical wires or decode raw basis strings.

Acceptance covers all 32 item-count combinations and the inclusive constraint
boundary. Two nonzero parameter sets and both objectives compare the exact
original and native programs with an independent Circuit using explicit RX
and classical cost tables. Full complex amplitudes agree within 1e-12, including
absolute phase and field order. Hidden work probability remains below 1e-24.
Qubit, operation, depth and memory budgets refuse before execution.

Observed resources are five qubits, zero work bits, 848 operations and depth721;
the original uses856 operations at the same width/depth. This is a local quantum
program counterpart. The complete associated Notebook workflow remains pending.
