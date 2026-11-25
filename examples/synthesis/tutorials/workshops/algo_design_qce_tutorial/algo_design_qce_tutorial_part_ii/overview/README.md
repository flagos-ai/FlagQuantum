# Native QCE algorithm-design tutorial part II

Open [walkthrough.ipynb](walkthrough.ipynb) from the repository root, or run:

```bash
python -m examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_ii.overview.main
```

The complete workflow retains all nine source code stages, with exercise placeholders
replaced by explicitly labeled reference solutions. It executes three-layer MaxCut,
slack-penalty Knapsack, and feasible-phase Knapsack, then repeats both Knapsack
solution executions. The source sizes remain three, nine and five logical qubits.
The frozen sibling quantum counterpart is unchanged.

The quantum slack phase uses coefficient2 on the squared equality penalty. Both
Knapsack optimizers use coefficient4 on the linear positive constraint violation.
These are intentionally different functions. The mixer uses RX(beta), not RX(2beta).
Native named a/b/slack ports preserve the source struct widths without a new API.

All angles bind at synthesis. The public native COBYLA optimizer replaces the
provider optimizer using exact local expectations. Actual objective calls are capped
at40 for MaxCut and50 for Knapsack. Native optimizer iterations are different units.
The selected parameters are the last actually evaluated query, matching the source
trace selection. Budget exhaustion is reported as nonconvergence. Optimization
trajectories and comparative convergence claims are not reproduced or promised.

Provider sampling defaults are made explicit as1000 local draws. Each Knapsack
final table preserves10 separate actual draws, including the repeated solutions.
Four figures show actual objective-call trajectories. Full independent Circuit
comparisons use two nonzero parameter vectors for each model, including global
phase, register/array bit order and zero extra compiler workspace.

See [WORKFLOW.md](WORKFLOW.md) for the semantic boundaries and
[provenance.json](provenance.json) for the pinned source-cell mapping. Source files
are provenance inputs only; the user workflow imports no Classiq code or Qmod.
