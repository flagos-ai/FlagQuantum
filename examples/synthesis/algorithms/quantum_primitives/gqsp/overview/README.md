# Complete GQSP phase-finding and sampling workflow

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.quantum_primitives.gqsp.overview.main
python -m examples.synthesis.algorithms.quantum_primitives.gqsp.overview.comparisons.circuit
```

The [Notebook](walkthrough.ipynb) retains the source coefficient scaling 0.99,
seven fractional input bits, Laurent shift three, actual phase finding,
quantum execution, 100000 samples and comparison plot. It reuses the existing
[native qfunc composition](../main.py), public numeric probabilities, NumPy
and Torch. Matplotlib is optional and imported only by plotting.
No source-language loader or cloud package installation is needed.

The [classical phase helper](phases.py) completes a real polynomial with a
spectral factor and strips its degree into native X/U/X/Z rotations. Its
conservative profile supports degree one through six, finite float64 inputs,
nonzero endpoints and coefficient L1 norm below one. It rejects ill-conditioned
roots or failed numerical identities; it has no fixed-phase fallback. This
is an example-local numerical routine, not a general production GQSP solver,
a realization theorem or an analytic approximation certificate.

The program reads fractional values through `program.numeric_probabilities`;
users need no physical wire numbers, integer scaling or bitstring decoding.
An explicit seed-123 multinomial draw preserves the original 100000-sample
experiment using the actual exact local distribution. This is local classical
sampling, not hardware execution or a runtime shot API. Exact probabilities
and finite-sample errors are checked separately. Indicator-zero probabilities
are divided by 0.99 squared and retain their joint normalization. Zero-count
bins remain visible. The curve label includes its actual 2*pi*x argument,
which the source label omits; the legend sits above the data.

Newly computed phase branches can change complementary complex amplitudes.
Full-state comparison uses an independent Circuit with the same new phases;
the [frozen original quantum program](../WORKFLOW.md) remains separately checked.
There is no claim of equivalent solver internals, optimal phase decomposition
or lower circuit cost than the frozen phase branch.

[Acceptance details](WORKFLOW.md). [Source provenance](provenance.json).
[Third-party license](../../../../CLASSIQ_LICENSE.txt).
