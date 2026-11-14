# Native kidney exchange optimization

The complete workflow retains three donors, three patients, nine binary
assignment bits, all original compatibility scores, five QAOA layers,
penalty factor two, width optimization, a 100-iteration limit and CVaR 0.7.
Each donor and each patient can participate in at most one match.

The native quantum function uses the original pair-conflict penalty: two
units for every conflicting pair within a donor row or patient column.
It does not add slack bits or silently replace the penalty with a squared
cardinality equality. The classical model is freshly evaluated for all
512 assignments, separately from quantum probabilities.

The public native Nelder-Mead optimizer and exact local CVaR replace the
platform's private optimizer and shot-based cost estimates. The initial
angle schedule is explicit. Each angle tuple is bound at synthesis time;
this does not claim runtime parameter binding or an identical optimization
trajectory. The complete run retains the original 100-iteration setting.
Final observations are actual seeded finite CPU samples.

Raw compatibility, conflict penalty, penalized cost and legal-matching
feasibility are reported separately. The source's final printed value
`-cost` is retained as a diagnostic: it is the raw compatibility only when
the conflict penalty is zero. The graph shows the best actually observed
penalized-cost assignment and labels its feasibility. A missing feasible
observation remains absent. An exhaustive legal-matching reference is
computed only after sampling and never repairs the sampled answer.

The source's classical solver is commented out; the native walkthrough
reports that no initial classical solve was performed. Model tables and
explicit arrays replace Pyomo printing. Native drawings replace the
platform viewer. Trace, weighted sample histogram, matching table and
bipartite graph preserve all original output roles. The original request
for no proprietary transpilation is represented by the existing native
checked gate construction without a platform-basis equivalence claim.

Run `walkthrough.ipynb` or:

```bash
python -m examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.overview.main
```
This remains a local correctness and workflow example, without cloud
execution, hardware timing, advantage or scalability claims.
