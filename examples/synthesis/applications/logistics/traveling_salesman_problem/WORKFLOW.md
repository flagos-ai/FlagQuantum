# Traveling salesman costs with native quantum functions

`visits[4 * city + slot]` is a Boolean assignment decision. The compiler handles
physical wire placement and reversible implementation. The native helper keeps
the retained four-city model's non-symmetric distance matrix, once-per-city and
once-per-slot squared constraints, and all eight QAOA layers.

```python
import flagquantum as fq
from examples.synthesis.applications.logistics.traveling_salesman_problem.main import build_program

params = tuple((-1) ** i * (0.004 + 0.001 * i) for i in range(16))
program = build_program(params)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("visits",))))
print(program.report["resources"])
```

The first eight parameters are cost angles, followed by eight RX mixer angles.
The original program contains only the three transitions between consecutive
slots. It does not include a final-to-first transition. The native program
preserves that detail rather than silently changing the application objective.

The independent Circuit reference enumerates classical two-bit edge factors
and four-bit constraint residuals. All 65,536 assignments are checked against an
independent classical objective, and two nonzero parameter sets exercise the
complete statevector in both optimization modes. Test-only original comparisons
also account for the four separately named source rows in declared readout order.
The native output exposes one named assignment array with logical bit zero at
city zero, slot zero. Hidden work probability must remain below 1e-24.

Observed default resources: sixteen qubits, 952 operations, depth 315 and no
workspace. The original source uses 929 operations and depth 157. These are
measured resources, not an optimality claim. Complete Notebook migration remains
pending.
