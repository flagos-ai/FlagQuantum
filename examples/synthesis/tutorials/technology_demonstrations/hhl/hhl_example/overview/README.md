# Original-sized HHL precision study

```sh
python -m examples.synthesis.tutorials.technology_demonstrations.hhl.hhl_example.overview.main
```

Open `walkthrough.ipynb` for the complete source workflow. Keep the original
rounded 8 by 8 matrix and RHS, precisions 2 through 8, reciprocal rotations,
postselection and squared overlaps. Actual operator-exact CX/U3 resources and
independent complete physical states are measured at every precision.

The original rounded RHS has norm 0.9999999994356988; preparation explicitly
normalizes it, while the classical solve uses the unchanged raw RHS. The source
overlap compares normalized solutions, so it is invariant to that scalar.
Historical Qiskit arrays remain labeled historical; optional source SDK cells
are entirely commented and are not represented as new SDK runs.

Outputs: `artifacts/development/hhl_precision/`. No source imports, compiler
changes or new public contracts. Frozen quantum-program examples are unchanged.
See `provenance.json` for reviewed source stages.
