# Hydrogen binding curve with native VQE

The [walkthrough](walkthrough.ipynb) builds the neutral STO-3G hydrogen
molecule at all twelve source distances from 0.20 to 3.50 angstroms.
Each restricted Hartree–Fock calculation produces a four-mode Hamiltonian.
The two paired determinants form an invariant one-qubit sector, with the
Hartree–Fock determinant at logical `|1>`.

The workflow evaluates the retained native one-angle ansatz on the local
simulator for every COBYLA query. Each solve starts at zero with a budget
of 500 optimizer iterations. The plot compares those measured statevector
expectations with the independently computed RHF and full CI energies.
The source MP2, CISD and CCSD energies do not enter the plot. The independent
chemistry audit also evaluates those three methods.

Run from the repository root with FlagQuantum and Matplotlib.

```sh
python -m examples.synthesis.applications.chemistry.molecular_energy_curve.overview.main
```
