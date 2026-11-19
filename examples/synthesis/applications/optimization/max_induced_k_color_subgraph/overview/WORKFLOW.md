# Source workflow mapping and acceptance

Pinned source: `applications/optimization/max_induced_k_color_subgraph/max_induced_k_color_subgraph.ipynb`
at `b69c4893d8264d1e59ce0909df2bd88efed25e00`.
`provenance.json` records the source hash and every reviewed code cell.

Use a deterministic circular graph layout in place of the source Kamada-Kawai
layout, which requires SciPy. This changes vertex positions only; graph edges,
node/color identities, objective and quantum state remain unchanged.

| Source cells | Native workflow |
| --- | --- |
| 3, 5, 7 | Six-node seed-7 Erdos-Renyi graph with probability 0.5 and eleven edges; two colors, no same-color edge, at most one color per node, maximize selected nodes. |
| 9, 11 | Reuse frozen native qfunc with eight layers and sixteen bound real angles; inspect actual resource report. |
| 13, 15 | Native Nelder-Mead, 50 iterations, exact local CVaR 0.7; plot every actual quantum evaluation. |
| 19, 21, 23 | 1000 optimized and 1000 uniform CPU samples; shared fifty-bin cost histogram. |
| 25 | Draw best actually observed feasible coloring; report missing feasible observations explicitly. |

The source directed adjacency quadratic counts every undirected edge twice.
Preserve the frozen minimization cost over the full 4096-assignment joint domain:

```text
-popcount(a | b) + popcount(a & b)
+ (2 * (same_color_edges(a) + same_color_edges(b)))**2
```

Each native cost layer applies exp(-i*gamma*cost), including constant global
phase. Mixers apply RX to each of twelve bits. This penalty is not the original
objective on infeasible assignments; check feasibility and maximize colored
nodes separately. Uncolored nodes are permitted. Both six-bit arrays are modeled
outputs, not temporary workspace. Zero workspace does not mean either word is zero.
Numeric bit i denotes graph node i; physical state axes are reordered separately.

Reuse native initial parameters `DEFAULT_PARAMS` and simplex step 0.07. Public
Nelder-Mead replaces the private optimizer; exact local probabilities replace
shot estimation during search. Each evaluation synthesizes and executes. The
50-iteration budget is retained, without claiming an identical trajectory.
Use actual CPU observations with seeds 2718 and 31415. Compute the separate
4096-assignment exhaustive reference only after sampling; it never supplies
parameters, probabilities or observed solutions. Report actual convergence,
feasible observation counts and any gap to the reference without claiming
optimality or quantum advantage.

Explicit bounds: twelve qubits, zero workspace, 40000 operations, 16 MiB state
planning. The original 34185-operation program refuses a 20000-operation limit.
State-planning memory is not total Python process RSS. Classification:
`single_device_fast_path`. All code uses native FlagQuantum.

Acceptance covers all 4096 independent costs, all absolute cost-layer amplitudes
at two nonzero gammas, two independent complete complex states, named readout,
all twelve numeric-bit/node mappings, 50-iteration optimization, real samples,
missing observations, invalid controls and resource refusals, actual Notebook
and CLI launches, figure inspection and repository checks. Coverage is credited
only after acceptance is complete; staging is not a completed Notebook.
