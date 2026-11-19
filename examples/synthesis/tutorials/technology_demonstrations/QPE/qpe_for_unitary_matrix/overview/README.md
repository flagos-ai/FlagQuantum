# Native matrix QPE across all source precisions

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.overview.main
```

The complete workflow recomputes the seeded8x8 matrix, runs every precision1..8,
checks native and CX/U3 complete states against an independent Circuit, validates
named phase marginals and plots actual resource counts. Historical Qiskit data
is labeled separately. [WORKFLOW.md](WORKFLOW.md) records the source's unused
matrix shift and the retained environment semantics.
