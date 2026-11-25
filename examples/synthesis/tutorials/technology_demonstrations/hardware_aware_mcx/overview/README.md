# Hardware-aware MCX synthesis

Run the complete native workflow:

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.hardware_aware_mcx.overview.main
```

[walkthrough.ipynb](walkthrough.ipynb) maps all three source code cells. One
15-control program is synthesized with a width ceiling of18, a depth objective
and a CX/U basis for all-to-all and line connectivity:

```python
import flagquantum as fq
from flagquantum.synth import (
    Output, QArray, QBit, X, allocate, control, qfunc, synthesize,
)

@qfunc
def main(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(15, cntrl)
    allocate(target)
    control(cntrl, lambda: X(target))

program = synthesize(
    main, objective="depth", max_qubits=18,
    basis=("cx", "u"), topology="line", memory_limit_bytes=512 * 1024**2,
)
print(program.report["resources"])
print(dict(program.probabilities(fq.run(program), registers=("target",))))
```

The original target stays zero because the controls start at zero. Supplementary
checks activate every control, start the target at both zero and one, leave one
control disabled, and use a complex coherent state. Every amplitude is checked
against independent Circuit preparation followed by the all-ones truth
permutation. This oracle shares no gate decomposition and avoids constructing
a65536-square matrix. The working bit returns to zero. Raw state comparisons
apply the final routing layout; named readout handles it automatically.

The source scenarios use17 physical bits, including one clean working bit.
The normalized all-to-all circuit has975 operations and depth681; the line
circuit has2461 operations and depth1314. These are static costs over the actual
native executable. The compiler uses the same exact one-clean decomposition
with target-specific routing, not the proprietary source engine's algorithms.
No optimality, equal-resource, hardware timing or seed-equivalence claim is made.
The existing larger-budget parent example remains unchanged. See
[WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json).
