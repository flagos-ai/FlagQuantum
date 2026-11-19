# Two-vehicle routing with native QAOA

Run [the walkthrough](walkthrough.ipynb) or the complete command from the repository root.

```bash
python -m examples.synthesis.applications.logistics.vehicle_routing_problem.overview.main --figures routing-figures
```

The workflow freshly generates the original six locations with RandomState(10), normalizes their Euclidean distances, and binds them to the native sixteen-bit, twelve-layer QAOA function. It keeps the original city/position constraints, objective normalization, mixer scale, and initialization schedule.

Every objective query uses 30,000 actual finite draws. The original SciPy transcript expands its requested one evaluation to 26 for the 24-dimensional simplex. This implementation explicitly budgets 26 queries with native COBYLA and its original radius 0.7, then selects the best actually evaluated parameters. The native optimizer has its own trajectory. A final 30,000-shot sample is sorted by the original cost, and its lowest-cost observed assignment supplies the routes and plot.

Independent checks compare two full complex states, all 65,536 cost values, and the 24 feasible classical routes. The classical reference is evaluated after sampling and never modifies the observed assignment.
