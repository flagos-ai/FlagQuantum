# Adiabatic quantum linear solver

Both original 4 by 4 systems and all three three-qubit evolutions. Generate H0 and H1 from the model, retain the 7/50, 7/50 and 10/100 time schedules, and compare all eight amplitudes and direct-solve overlaps.

Run from the repository root with `python -m examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.overview.main`. The [Notebook](walkthrough.ipynb) includes the spectra and direct solution comparisons.

The native first-order formula preserves the Pauli mask order recorded in the retained program. The historical transpiler order and basis are unpinned, so the Notebook reports actual native amplitudes, overlap and static instruction depth. H1 uses the original matrix scale, while the direct solve uses its normalized inputs. The reported overlap includes population in both ancilla sectors.

[Source provenance](provenance.json).
