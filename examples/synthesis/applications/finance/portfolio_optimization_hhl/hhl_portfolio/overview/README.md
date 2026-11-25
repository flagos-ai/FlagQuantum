# HHL portfolio reconstruction from the original price data

Rebuild the source four-asset model from all 251 bundled observations. The
native program preserves the Hermitian embedding, six phase bits, 64 reciprocal
rotations, source input padding, and indicator-one and phase-zero readout.
Compare the entire complex state against an independent Circuit, then apply
the source bilinear fit to reconstruct the observable portfolio weights.

Run from the repository root with
`python -m examples.synthesis.applications.finance.portfolio_optimization_hhl.hhl_portfolio.overview.main`.
The [Notebook](walkthrough.ipynb) walks through the original stages and the
sample-free HHL reconstruction. It labels the weights with the actual price
column order. The CSV contains only the input observations from the source.
Returns, covariance, matrices, rotations, weights and plotted series are
recomputed locally.

[Source provenance](provenance.json).
