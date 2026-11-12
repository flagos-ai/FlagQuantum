# Max Clique workflow

Run from the repository root:

```bash
python -m examples.synthesis.applications.optimization.max_clique.overview.main
python -m examples.synthesis.applications.optimization.max_clique.comparisons.circuit
```

Open `walkthrough.ipynb` for the native qfunc, original seeded graph, actual
three-layer quantum optimization, local samples and workflow plots. The shared
example helper reuses the public Nelder-Mead optimizer; every objective call
synthesizes and executes the native program. Parameters are radians.

The 50-iteration bound and quantile 0.7 are preserved. Report convergence
honestly and filter feasibility before selecting sampled answers. The exact
bounded classical reference never supplies quantum optimization results.

[Workflow mapping and limits](WORKFLOW.md) disclose native replacements for
foreign modeling, optimization, sampling, classical solving and graph layout.
The unchanged native quantum program and independent Circuit live one level up.
