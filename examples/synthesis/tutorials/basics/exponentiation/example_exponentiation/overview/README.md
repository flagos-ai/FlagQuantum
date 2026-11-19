# Water Hamiltonian exponentiation

The [walkthrough](walkthrough.ipynb) computes the STO-3G water integrals and
the restricted Hartree–Fock orbitals locally. It freezes the first doubly
occupied spatial orbital and retains six active orbitals with eight
electrons. The native calculation fixes the phase of each real orbital by
its largest atomic-orbital coefficient. The Jordan–Wigner mapping orders all
alpha modes before all beta modes. A numerical cutoff removes cancellation
residue below the precision of the retained 551 Pauli terms.

The parent data file supplies the source term order and a coefficient
cross-check. The synthesized circuit uses the fresh coefficients for every
term. The example executes one first-order product step at time one,
compares the full complex state with an independent Circuit and reports the
resources after a checked CX/U3 conversion. The figure shows the
measured CX and U3 counts for the complete product step.

Run from the repository root with FlagQuantum and Matplotlib.

```sh
python -m examples.synthesis.tutorials.basics.exponentiation.example_exponentiation.overview.main
```
