# Hydrogen molecular eigensolver

The [walkthrough](walkthrough.ipynb) runs the selected source molecule,
neutral singlet hydrogen at 0.735 angstrom in STO-3G. It retains the alternate
geometries for reference. The native solver recomputes the full four-mode
Hamiltonian and the invariant two-determinant sector used by the original
tapered UCCSD example.

The hardware-efficient ansatz contains three X, RY and chain-CX layers.
The resource profile uses twelve nonzero angles, 0.07 through 0.84 radians.
The one-qubit UCCSD optimization starts from zero and has a 200-iteration
ceiling. At each evaluation the simulator supplies the probabilities for the
two nonconstant Pauli terms. The workflow draws one million measurements per
basis in aggregate and compares the finite-shot result with the exact
expectation at the returned angle and the full CI energy.

Run from the repository root with FlagQuantum and Matplotlib.

```sh
python -m examples.synthesis.applications.chemistry.molecule_eigensolver.overview.main
```
