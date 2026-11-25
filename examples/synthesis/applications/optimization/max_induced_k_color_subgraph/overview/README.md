# Native induced two-color subgraph Notebook

Run from the repository root:

```bash
python -m examples.synthesis.applications.optimization.max_induced_k_color_subgraph.overview.main
python -m examples.synthesis.applications.optimization.max_induced_k_color_subgraph.comparisons.circuit
```

Open `walkthrough.ipynb` for the full native workflow. Keep the original six-node
Erdos-Renyi graph (p=0.5, seed 7), two colors, eight layers, sixteen angles,
50 optimizer iterations, CVaR 0.7, and 1000 optimized plus 1000 uniform samples.
The original-size local search can take about 40 minutes on the measured CPU;
this is not a ten-minute introduction. Use the adjacent native quantum program
for the shortest synthesis and readout path.
For a deliberately shorter optimizer smoke run, pass `--maxiter 1` to the
workflow command. The default remains 50; a short run does not certify the
original iteration budget.

Maximize the number of colored nodes with no double-colored node and no edge
joining equally colored nodes. Nodes may be left uncolored. Read both six-bit
words by name; numeric bit i denotes node i. Report actual feasibility separately
from the retained quantum penalty. The best feasible observation stays `None`
if none was sampled. A separate exhaustive reference is computed after sampling
and never supplies optimizer answers or observations.

The native optimizer uses an explicit sixteen-angle starting point and step
0.07. Exact local probabilities replace private shot-based objective estimates;
optimizer trajectories are not claimed identical. The 16 MiB state-planning
limit is not total process memory. The program uses twelve qubits, zero temporary
workspace and an explicit 40000-operation ceiling. This workflow requires no
foreign quantum source loading, remote services or credentials.

See [source mapping and acceptance](WORKFLOW.md) for precise semantics and limits.
