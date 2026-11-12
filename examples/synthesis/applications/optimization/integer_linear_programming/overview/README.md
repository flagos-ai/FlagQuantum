# Native integer linear programming Notebook

Run the complete workflow and independent Circuit comparison from the repository root:

```bash
python -m examples.synthesis.applications.optimization.integer_linear_programming.overview.main
python -m examples.synthesis.applications.optimization.integer_linear_programming.comparisons.circuit
```

Open `walkthrough.ipynb` for the original integer model, native three-layer
quantum function, explicit synthesis budgets, actual CVaR optimization,
1000 optimized and 1000 uniform samples, and plots. Maximize x0+2*x1+3*x2
with each variable in {0,1,2,3}; check all three original inequalities explicitly.
The source files and Notebook use native FlagQuantum code without loading
foreign quantum sources.

Preserve the frozen eight-bit quantum encoding. Three two-bit integer registers
and two modeled slack outputs define a 256-assignment joint distribution. The
slack outputs are not temporary workspace and need not be zero. Report original
objective and feasibility separately from the retained quantum penalty cost.
If no feasible assignment is actually observed, report `None`.

Public native Nelder-Mead and exact local CVaR replace the source private
optimizer and its shot estimator during search. Keep three layers, the
90-iteration setting, quantile 0.7 and actual final seeded observations. The
bounded 64-assignment classical reference replaces Couenne only after sampling;
it never supplies optimizer inputs or observed answers. Report convergence and
feasible sample counts without claiming quantum advantage.

[Source-cell mapping and limits](WORKFLOW.md) documents the original model,
retained quantum cost, replacements and acceptance criteria.
