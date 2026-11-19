# Fermionic Gaussian preparation

Original four-mode orbital factorization, eight-mode one-electron ring, two-mode pairing Hamiltonian, Majorana canonical planes, Bogoliubov relation, and full-state validation across all 25 code stages.

Run from the repository root: `python -m examples.synthesis.algorithms.quantum_state_preparation.fermionic_gaussian.overview.main`. The [walkthrough](walkthrough.ipynb) includes the independent comparisons.

The ring uses freshly computed adjacent orbital rotations from the native Fermi–Hubbard example. The two-mode case uses Hermitian iA eigenvectors to reconstruct the real canonical planes, then solves the two quasiparticle annihilator equations and synthesizes a bounded parity-preserving preparation. Direct Fock-space diagonalization is an independent oracle only. The final matrix check verifies the canonical unitary relation; it does not reproduce the OpenFermion double-Givens or particle-hole factor order. This example makes no general multi-mode pairing claim.

See [provenance](provenance.json).
