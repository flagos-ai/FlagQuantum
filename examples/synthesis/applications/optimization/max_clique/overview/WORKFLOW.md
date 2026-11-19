# Source workflow mapping

Pinned source and complete reviewed code-cell hashes appear in `provenance.json`.

| Source cells | Native workflow |
| --- | --- |
| 3, 5 | Native graph/model inspection replaces Pyomo; same seven-node graph,p0.6,seed79 and complement constraints. |
| 7, 9 | Reuse unchanged three-layer native qfunc; inspect actual resources and execute locally. |
| 11, 12 | Actual bounded50-iteration,quantile0.7 native optimization and cost trace. |
| 16, 18, 20 | 1000 optimized and1000 uniform local samples; empirical unpenalized cost histogram. |
| 22, 23 | Largest observed feasible clique and graph highlights. |
| 26, 27 | Exact bounded128-assignment feasible enumeration replaces Couenne; separate reference graph. |

The existing native qfunc remains unchanged. The independent Circuit verifies
absolute complex amplitudes and bit order at two nonzero parameter bindings.
Native synthesis uses no dense reference matrix and requires zero workspace.

The source private Pyomo conversion and optimizer are not executed. Explicit
native lower-tail penalized CVaR0.7 and public Nelder-Mead replace the optimizer;
the original layer and iteration limits remain. The initial six radian angles
are `(0.13,-0.27,0.19,0.22,0.31,-0.17)` and initial simplex step is0.07.
Every evaluation synthesizes/runs the actual native program and reads named
probabilities. A bounded search may remain unconverged. No foreign trajectory,
sample values,gate counts,hardware results or convergence parity is claimed.

Optimized CPU sampling uses seed2718; uniform CPU sampling uses seed31415.
Each draws1000 observations. Penalized minima can include infeasible words;
filter graph feasibility before choosing the best observed subset. Histograms
show negative unpenalized subset size over all observations,so the lowest bar
is not automatically a valid answer. Exact feasible enumeration is a separate
labeled reference calculated after search and sampling,not an optimizer oracle.

Seeded force layout replaces source Kamada-Kawai without adding SciPy. All
source figure roles remain: original graph,actual cost trace,sampled cost
comparison,observed feasible graph and exact classical reference graph.
Runtime classification: `single_device_fast_path`.

The retained cost is `-sum(v)+4*conflicts**2`,where conflicts count unordered
missing edges. The complement adjacency quadratic form counts each unordered
pair twice; squaring it gives the retained factor4.

Array element `v[i]` is bit i of the named packed integer, so node0 is the
least-significant bit. `selected_nodes` performs this decoding for users.
Full-state Circuit references retain their physical axis order independently;
classical model tables and sample node labels use the declared array indexing.
