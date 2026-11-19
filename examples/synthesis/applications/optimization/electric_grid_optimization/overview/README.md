# Native electric-grid optimization

Run `python -m examples.synthesis.applications.optimization.electric_grid_optimization.overview.main` from the repository root, or execute `walkthrough.ipynb`.

The original problem has three sources and four consumers, twelve assignment bits and six retained slack bits. Every source supplies at most two consumers and every consumer has exactly one source. Both bits in each slack register have weight one. Four cost/mixer layers and the original 100-iteration mean-cost budget are retained.

Public native Nelder-Mead and exact local expectation values replace proprietary shot-based optimization. Every parameter tuple is synthesized anew. Final optimized and zero-angle uniform distributions produce actual finite CPU samples. Raw supply cost, residual penalty and decision feasibility remain separate, including infeasible observations. An exhaustive constrained classical reference is solved only after sampling; it replaces the external Couenne stage without using its answer to repair observations.

All thirteen source stages map to the Notebook. See [WORKFLOW.md](WORKFLOW.md) for acceptance scope and [provenance.json](provenance.json) for source pins. The existing parent quantum program and Circuit comparison stay frozen. These examples make no performance, hardware or scalability claim.
