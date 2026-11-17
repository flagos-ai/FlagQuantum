# Two-asset Rainbow option with direct exponential loading

Regenerate the original Gaussian bins, Cholesky coefficients, affine maximum rounded
to quarter increments and strike threshold. The native quantum program performs the affine
maximum, controlled rotations and reversible reflection. The two original
asset words and the five auxiliary bits remain part of the checked state.

Run from the repository root with
`python -m examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.overview.main`.
The [Notebook](walkthrough.ipynb) walks through the fresh model, sampling and
the resulting estimate. The local estimator uses the source precision and
confidence inputs and caps the Grover power at four. It reports the resulting
finite confidence interval. The independently enumerated direct-grid payoff
and the historical source reference are shown as distinct comparisons.

[Source provenance](provenance.json).
