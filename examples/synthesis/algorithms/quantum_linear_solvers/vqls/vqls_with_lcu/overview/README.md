# Variational linear solve with LCU

All 17 source code stages, the original three-term 8 by 8 matrix, three system and two selector bits, all nine ansatz angles, the conditional cost with 204800 actual draws per query, a cap of 2000 objective queries, and comparison of the final state with the direct solve.

Run from the repository root: `python -m examples.synthesis.algorithms.quantum_linear_solvers.vqls.vqls_with_lcu.overview.main`. The [Notebook](walkthrough.ipynb) includes the finite query trace and direct comparison.

Native COBYLA replaces the unpinned hosted optimization, with seed 1000 for the source initial angles and an independent local sampling stream. Preserve the last executed parameter binding returned by the source wrapper. Statevector readout yields the actual final amplitudes. The final phase is aligned to the source last-component convention. The routine counts postselection events and reports the actual trace; no historical optimizer trajectory, convergence, or speedup claim.

[Provenance](provenance.json).
