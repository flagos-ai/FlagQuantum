# Source workflow mapping and acceptance

Pinned source: `applications/optimization/min_graph_coloring/min_graph_coloring.ipynb`
at `b69c4893d8264d1e59ce0909df2bd88efed25e00`.
`provenance.json` records the source hash and all reviewed code cells.

| Source cells | Native workflow |
| --- | --- |
| 5, 7, 9 | Original five-node seed-79 graph, edges (0,1), (0,3), (0,4), (1,3), (2,3); three available colors; exactly one color per vertex; no same-color edge; minimize used colors. |
| 11, 13 | Frozen native qfunc, fifteen modeled bits, six layers and twelve angles; actual resource report. |
| 15, 17 | Native Nelder-Mead, 100 iterations, exact local CVaR 0.7; actual evaluation trace. |
| 20, 22, 24 | 1000 optimized and 1000 uniform observations; common fifty-bin cost histogram. |
| 26, 27 | Draw best actually observed feasible coloring; explicit missing-observation report. |

Preserve the minimization cost over all 32768 joint assignments:

```text
used_colors + 10 * sum((colors_per_vertex - 1)**2)
+ 40 * same_color_undirected_edge_count**2
```

The source counts every undirected edge twice before squaring. Each cost layer
applies exp(-i*gamma*cost), including constant global phase; mixers apply RX on
all fifteen bits. Feasibility and the number of used colors remain separate
from penalty costs. The fifteen modeled outputs are not temporary workspace.
Numeric bit `5 * color + vertex` identifies each pair; tensor axes are reordered
separately. The domain contains 243 exactly-one assignments and 24 proper
colorings. The triangle (0,1,3) requires three colors; the separate exhaustive
reference is computed after sampling and never supplies observed answers.

Reuse `DEFAULT_PARAMS` and simplex step 0.07. Public Nelder-Mead replaces the
private optimizer; exact probabilities replace shot-based objective estimates.
Each evaluation synthesizes and executes the actual native program. Preserve
the original 100-iteration budget without claiming the same search trajectory.
Final CPU sampling uses seeds 2718 and 31415. Report actual convergence and
feasible counts without claiming optimality or quantum advantage.

Use deterministic circular graph layout instead of the source Kamada-Kawai
layout to avoid an implicit SciPy dependency; only drawing positions change.

Classification: `single_device_fast_path`. Bounds: fifteen qubits, zero workspace,
40000 operations and 32 MiB state planning. The native program has 19975
operations and depth 13069; the frozen original replay has 17881 operations and
depth 13284. These are distinct measurements, not resource equality.

Acceptance covers all 32768 independent classical costs, two complete absolute
cost-phase vectors, two full independent complex states and named readout, all
fifteen numeric bit mappings, original-budget optimization and real observations,
missing observations, invalid controls, resource refusals, actual Notebook/CLI
launches, figure inspection and repository checks. Coverage is credited only
after acceptance is complete.

## Recorded local acceptance outcome

The full native run completed 100 iterations and 201 actual evaluations without
convergence. CVaR decreased from 464.23276516209427 to 304.1676004498076. Both
1000-observation sets contained zero feasible colorings; `best_feasible` remained
`None`. The separate exhaustive reference was numeric word 1240, using three
colors. It was not inserted into observations or the displayed coloring. This
acceptance certifies the implemented workflow and its verified quantum semantics;
it does not establish successful optimization or a quantum advantage.

The recorded initialization and exact local objective differ from the private
source optimizer; these outcomes are specific to this native local run. The
fresh kernel executed all five clean code cells and three full-workflow scenario
bodies in two evidence-only cells. Another 48 focused pytest cases passed. The
complete default retained the prior 24 failures and 6 errors, with 19619 passing
tests; no failure/error nodes were added or removed.
