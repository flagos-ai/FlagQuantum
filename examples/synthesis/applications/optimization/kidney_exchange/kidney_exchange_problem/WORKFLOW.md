# Kidney exchange with native quantum functions

The program prepares nine Boolean matching decisions, applies the complete five
QAOA layers, and leaves every register and work bit to the compiler. Application
index `3 * donor + patient` addresses a matching decision; it is not a physical
wire. The cost helper rewards the original nine utilities and penalizes every
pair sharing a donor or a patient by two. It does not introduce cycle constraints
or change the retained model into a different kidney-exchange formulation.

```python
import flagquantum as fq
from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.main import build_program

program = build_program(tuple(0.01 * (i + 1) for i in range(10)))
result = fq.run(program)
print(dict(program.probabilities(result, registers=("matches",))))
print(program.report["resources"])
```

The first five parameters are cost angles, followed by five RX mixer angles.
Use `main` with `synth.synthesize` to set additional physical resource limits.

Both nonzero parameter sets in `comparisons/circuit.py` are checked in full
complex amplitudes, including absolute global phase. The Circuit reference uses
classical conflict detection and independent truth factors. Test-only source
comparisons separately permute the original nine scalar field outputs into
their declared readout order. Native array index zero denotes donor zero,
patient zero; no caller decodes simulator bit strings.

Observed default resources: nine qubits, 183 operations, depth 74 and no workspace.
The retained source uses 180 operations and depth 39. This migration demonstrates
semantic equivalence and readable native intent, without claiming resource
optimality. The associated Notebook's complete workflow remains pending.
