# Set Cover

Choose as few subsets as possible while covering all ten elements.

All fourteen source stages, eight original subsets covering ten elements, eight decision bits and fifteen encoded slack bits, three QAOA layers, penalty 10, sixty optimizer iterations, exact lower-tail CVaR at 70%, finite optimized and uniform samples, and independent exhaustive constrained comparison.

From the repository root, run:

```bash
python -m examples.synthesis.applications.optimization.set_cover.overview.main
```

See [the Notebook](walkthrough.ipynb) for the full model, sample distributions, and exact comparison. [Provenance](provenance.json) identifies the original and documents the adaptations.

Reuse the native qfunc for the original subset table. Sum costs over the complete 23-bit distribution using a bounded integer energy histogram. Local Nelder–Mead with an explicit linear pi schedule replaces the unpinned hosted optimizer. Use 1000 actual CPU draws per comparison. Select the least penalized observed joint word without repairing it. An independent exhaustive search over 256 decision words replaces Couenne after sampling. No historical trajectory equivalence or optimality guarantee.
