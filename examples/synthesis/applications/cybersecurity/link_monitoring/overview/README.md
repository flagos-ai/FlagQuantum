# Native link monitoring

Complete original 17-cell wireless sensor link-monitoring workflow: all 12 resources, 13 edges, node 1 fixed to one, one QAOA layer, penalty 2, CVaR 0.9, native 60-iteration search, actual 1,000 draws, observed minimum feasible cover and original/solution network views.

The source fixes resource node 1 in its classical model before synthesis.
Its captured quantum export leaves this node free, so the complete native
Notebook uses a separate qfunc: prepare node 1 with X, prepare the other eleven
nodes uniformly, retain the full cover cost and mix only the free nodes. All
twelve logical nodes remain explicit. The previously verified 216 quantum
counterparts, including the unfixed export, stay unchanged.

Direct Python cover cost and exhaustive constraint diagnostics replace Pyomo.
Penalty 2 matches all 21 captured Observable coefficients for all 4,096 words;
constant phase, bit order and fixed-node invariant are independently checked.
Native Nelder-Mead, explicit initial angles (0.13, -0.27), exact local CVaR 0.9
and 1,000 seeded actual draws replace implicit provider optimizer/session
choices. Preserve the 60-iteration budget and report real iterations,
evaluations and convergence. No hosted trajectory equivalence is claimed.
Select the smallest observed feasible cover; retain infeasible samples and do
not inject an exhaustively known answer. Spring layout replaces Kamada-Kawai
without adding SciPy. No remote jobs, release, hardware or advantage claim.

Run from the repository root:

```bash
python -m examples.synthesis.applications.cybersecurity.link_monitoring.overview.main
```

Open `walkthrough.ipynb` for all seventeen original teaching stages,
actual search and samples, network plots and independent Circuit checks.
See [WORKFLOW.md](WORKFLOW.md) for acceptance boundaries.
