# Complete source workflow acceptance

Review all eight executable source cells: phase-finder installation, scaled
Laurent coefficients, negative power three, diagonal phase generator, seven
fractional input bits, synthesis/viewer, 100000-shot sampling and plot. Replace
the cloud phase service with actual bounded local spectral factorization,
not precomputed phase rows. Native resource reports replace the cloud viewer.
No production API or dependency is added.

Validate the complex polynomial response on 4097 points, spectral coefficient
identities and degree stripping, including other coefficient scales, signs
and parity. Invalid widths, endpoints, dtypes, nonfinite values and coefficient
norms refuse. Ill-conditioned or failed numerical completion must not silently
produce an accepted phase sequence. Native width, operation, depth and memory
budgets remain enforced.

Compare the complete newly phased state to the independent Circuit without
phase alignment, including logical x/indicator order and zero work. Read all
256 fractional/indicator outputs through numeric_probabilities. Verify exact
indicator-zero joint probabilities against 0.99^2*cos^6(2*pi*x)/128, then perform
100000 local multinomial samples and check finite-sample errors separately.
Do not normalize the selected block to probability one. Keep zero-count bins.
New/frozen phase branches are separate: the frozen quantum counterpart keeps
its original full-state contract, and the full Notebook keeps its projected
polynomial and sampling contract.

Execute the four actual Notebook code cells and both CLI modules. Check all
128 plotted values, sample counts, target values, labels and axes, and inspect
the rendered Jupyter figure. Correct the source's omitted 2*pi argument in the
legend and move the legend above the data; retain x and f(x) axes. Saved source
Notebook outputs must be empty. No hardware, general solver, optimality or
analytic uniform-error claim is made.

```sh
python -m pytest tests/integration/test_synthesis_gqsp_notebook.py tests/integration/test_synthesis_native_gqsp.py -q
```
