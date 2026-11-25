# Native HHL with exact and powered Trotter evolution

Run both complete original routes:

```bash
python -m examples.synthesis.algorithms.quantum_linear_solvers.hhl.overview.main
```

[walkthrough.ipynb](walkthrough.ipynb) preserves all 14 original code stages,
the dense four-by-four matrix, normalized right-hand side and four-bit QPE.
It recomputes the Pauli expansion, synthesizes exact and power-dependent
first-order Trotter evolution, verifies complete absolute Circuit states and
converts both programs to an exact CX/U3 basis before reporting resources.
The final readout performs actual postselection, source global-phase correction,
source real projection, fidelity and both solution plots.

[WORKFLOW.md](WORKFLOW.md) explains the diagnostic and support boundaries.
