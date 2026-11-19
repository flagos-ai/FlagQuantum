# Executing parameterized quantum programs

Open [walkthrough.ipynb](walkthrough.ipynb) or execute the complete local
tutorial.

```bash
python -m examples.synthesis.tutorials.basics.quantum_programming.execution_tutorial_part2.overview.main
```

The circuit keeps one symbolic angle. The tutorial binds it to the
original single values and all 50 points from zero to twice pi. Sampling
uses actual 2,048-shot draws. The four-term Bell projector is evaluated
with finite draws in its shared Bell eigenbasis and checked against the exact
local expectation. An independent analytic amplitude formula
checks every bound state.

The local executor submits sampling and observation on the same bound
case, writes their actual results as immutable JSON records and retrieves
both records by content ID. The final two-parameter Bell exercise starts
from zero and uses the native COBYLA optimizer with the original
200-query ceiling. It reports the last evaluated result, the best
observed fidelity and the optimizer status.
