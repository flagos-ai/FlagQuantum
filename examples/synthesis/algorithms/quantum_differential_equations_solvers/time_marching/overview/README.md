# Magnus time marching

Run the full native walkthrough from the repository root.

```sh
python -m examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.overview.main
```

[The Notebook](walkthrough.ipynb) integrates the original four-coordinate
diagonal system from time zero to two, with four long slices and four
left-endpoint quadrature samples per slice. The classical reference solves
the diagonal system analytically at the original 100 plotting times.

Each slice regenerates the coefficients and phases of the degree-six even and
degree-seven odd exponential responses. The amplified path also computes a
globally bounded degree-seven fit to the original target slope on
`[-1/2, 1/2]`. An explicit phase-convention conversion maps these
fresh sequences to FlagQuantum's composed rotation blocks. The implementation
uses native qfunc for the Magnus steps and exact packed simulator operators
for the projector phases and conditional counter decrements.

Both integrators retain the original 11-bit layout and counter initialized at
four. Each circuit is checked against an independent complete tensor state
and sampled one million times. Probabilities from those samples are
postselected by counter equality and square-rooted to recover the source
amplitude estimates. Exact success probabilities remain distinct from
observed counts. The masked projector and permutation matrices are counted
as simulator instructions without hardware gate estimates.

[Source mapping](provenance.json) · [Third-party license](../../../../CLASSIQ_LICENSE.txt)
