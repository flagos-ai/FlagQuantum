# Native minimum graph coloring Notebook

Run from the repository root:

```bash
python -m examples.synthesis.applications.optimization.min_graph_coloring.overview.main
python -m examples.synthesis.applications.optimization.min_graph_coloring.comparisons.circuit
```

Open `walkthrough.ipynb` for the complete native workflow. Preserve the original
five-node Erdos-Renyi graph (p=0.3, seed 79), three available colors, six layers,
twelve angles, 100 optimizer iterations, CVaR 0.7, and 1000 optimized plus 1000
uniform observations. This original-size CPU search is not a ten-minute
introduction; use the adjacent native quantum program for the shortest synthesis
and readout path. Pass `--maxiter 1` for an explicitly shorter launch smoke test;
that run does not certify the original budget.

Minimize the number of used colors, assigning exactly one color to each vertex
and different colors to adjacent vertices. Read the fifteen-bit `colors` word by
name; numeric bit `5 * color + vertex` represents one vertex/color pair. Report
actual feasibility separately from the retained quantum penalty. A missing
feasible observation stays `None`. The exhaustive reference runs only after
sampling and never supplies optimizer answers or observations.

The native optimizer uses an explicit twelve-angle starting point and simplex
step 0.07. Exact local probabilities replace private shot-based estimates;
optimizer trajectories are not claimed identical. Bounds are fifteen qubits,
zero temporary workspace, 40000 operations and 32 MiB state-planning memory.
State-planning memory is not total process RSS. No remote service, credentials
or foreign quantum source loader is required.

See [source mapping and acceptance](WORKFLOW.md) for semantics and limits.

The recorded full local run did not converge or sample a feasible coloring.
See the acceptance outcome in `WORKFLOW.md`; complete workflow coverage does not
mean this optimizer found a solution.
