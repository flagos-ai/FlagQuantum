# One-dimensional Fermi–Hubbard dynamics

All 39 source stages: four-mode factorization, four-site eight-mode Slater preparation, quarter-filling interacting dynamics on steps 0–9, one-electron noninteracting dynamics and half-filling vanishing hopping on steps 0–10, and two 400-point classical one-electron comparisons.

From the repository root:

```bash
python -m examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.overview.main
```

See [the Notebook](walkthrough.ipynb) for state preparation, charge and spin dynamics, and the two limiting cases. [Provenance](provenance.json) describes the source and local substitutions.

Fresh real eigendecomposition and adjacent row rotations replace the OpenFermion factorization for these number-conserving real matrices. The occupied Slater determinant is checked independently from its minors. Native rotations and Pauli exchange preserve the source little-endian mode order, the implemented positive contact phase and the five Trotter stages. Each time point binds a static step count; exact named-state occupation expectations replace hosted finite-shot observe calls. The spread plot retains the final source formula with center (N + 1)/2. Exact expectations do not reproduce the original shot noise or hosted trajectory.
