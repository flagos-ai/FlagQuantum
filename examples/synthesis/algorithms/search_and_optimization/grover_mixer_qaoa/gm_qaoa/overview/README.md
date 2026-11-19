# Grover mixer compared to ordinary QAOA

All 15 source stages, four qubits and three layers in both the ordinary and Grover-mixer circuits, two budgets of 30 objective queries with 1000 actual draws per query, two independently sampled 1000-draw final distributions, and side-by-side comparison.

From the repository root:

```bash
python -m examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.overview.main
```

Open [the Notebook](walkthrough.ipynb) to inspect the costs, finite objective traces and sampled one-hot outcomes. [Provenance](provenance.json) identifies the original source and local adaptations.

Use native COBYLA with the source linear pi schedule and a hard cap of 30 actual objective queries for each run. All objective means use actual local draws with explicit advancing seeds. On budget stop retain the best evaluated parameters. The ordinary run uses the source squared one-hot penalty; the Grover mixer retains the Dicke preparation and literal controlled rotation and uses the unpenalized objective. No hosted execution or historical optimizer trajectory equivalence.
