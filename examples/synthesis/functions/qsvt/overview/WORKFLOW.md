# Complete source workflow acceptance

Retain all executable source stages: degree five, RandomState seed 1, odd parity,
500-point normalization, phase solving, native quantum program, execution,
selection of ind=0 and qsvt_aux=0 with ref summed for odd degree, sixteen projected
values, target values, and the green-scatter/target-line plot and axes. The source
0.02 readout tolerance is retained as context; native acceptance requires 1e-7.
No source-language loader or original code is included in the user examples.

Reuse existing native qfunc composition, classical angle binding, Nelder-Mead,
Torch, and named probability readout. The original classiq[qsp] installation cell
is replaced by an explicit existing local optimizer, not a package installation.
There are no production changes or hardware jobs. Matplotlib is optional and is
imported only for plotting; numerical acceptance still runs without it.

Three checks remain distinct. The frozen original quantum program retains its
fixed-angle full-state contract. Newly fitted angles match an independent Circuit
using the same new angles, with absolute phase, logical order and clean work.
New projected output magnitudes match the original target polynomial. Different
phase solutions can change complementary amplitudes, so full-state equivalence
across phase branches is not asserted.

Calibrate the scalar response against fixed angles only in tests. Runtime phase
fitting must never use those angles as its initial guess, answer or fallback.
Check endpoint half-angle convention, odd parity, source coefficients, polynomial
recurrence, fit quality and the actual quantum result. Invalid domains, array
shapes, nonfinite values, fit budgets and failed fits must refuse. Quantum width,
operation, depth and memory limits must fail closed. No guarantee of arbitrary
polynomial realizability, optimizer optimality or uniform analytic accuracy is made.

Execute the actual Notebook and both CLI launchers. Independently inspect plot
points, labels and series, and visually inspect the Jupyter figure. Source codecells
are hash-bound in provenance; saved Notebook outputs remain empty.

```sh
python -m pytest tests/integration/test_synthesis_qsvt_notebook.py -q
```
