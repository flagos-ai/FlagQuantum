# Fit and execute a QSVT polynomial

Run from the repository root:

```sh
python -m examples.synthesis.functions.qsvt.overview.main
python -m examples.synthesis.functions.qsvt.overview.comparisons.circuit
```

The [Notebook](walkthrough.ipynb) retains the source's degree-five odd polynomial,
seed 1, normalization over 500 points, phase finding, quantum execution and plot.
It reuses the [native qfunc composition](../main.py) with newly fitted angles.
Its named probability readout needs no physical wire numbers or bitstring decoding.
Classical phase values are bound at synthesis time, not runtime tensor parameters.

The existing native Nelder-Mead optimizer minimizes a classical two-dimensional
scalar-response model at 64 points. At most four seeded starts, each with at most
2500 iterations, are allowed. A separate 4097-point grid must agree within 1e-7;
failure raises instead of falling back to fixed angles. This is a numerical check
of this concrete six-angle profile, not a generic QSP solver or an analytic
uniform-error certificate. The local optimizer is not claimed equivalent to the
source's cloud phase finder. No new dependency is installed from the Notebook.

Native projector angles are twice the scalar QSP phases. Different phase solutions
can implement the same projected polynomial while changing complementary output
amplitudes. The frozen fixed-angle quantum-program example remains separate.
Full-state Circuit acceptance uses the newly fitted angles; source workflow
acceptance checks the projected polynomial values and original comparison plot.

[Acceptance details](WORKFLOW.md). [Source provenance](provenance.json).
[Third-party license](../../../CLASSIQ_LICENSE.txt).
