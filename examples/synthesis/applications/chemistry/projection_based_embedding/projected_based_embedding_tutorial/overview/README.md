# Projection-based embedding

Run the complete original cc-pVDZ water workflow with optional local PySCF:

```bash
python -m examples.synthesis.applications.chemistry.projection_based_embedding.projected_based_embedding_tutorial.overview.main
```

Keep all24 atomic orbitals, ten electrons, oxygen as the fragment, three concentric-localized virtual orbitals, eight active spatial orbitals,14 tapered bits and30 spin-preserving singles. The21-stage Notebook runs fresh B3LYP, source-defined virtual selection, four embedding validations, native mapping and100 actual optimizer queries. The source technical notes explicitly define the coupling `C_virtual.T @ S @ F @ C_occupied_fragment`; no archived integrals or fitted energies select the orbitals.

All five occupied orbitals belong to the original oxygen fragment. The environment density, embedding potential, level-shift projector and environment correction therefore vanish exactly. Physical and embedded Hamiltonians coincide; the native workflow preserves both roles and reuses their identical encoding. This example is scoped to that original empty-environment case and refuses a changed occupied partition.

The native canonical fermion representation combines normal-ordered terms. Native interleaved Jordan-Wigner mapping and two spin-parity symmetries define the actual fourteen-bit Clifford frame. The source0.1 coefficient threshold is applied in that declared frame. Native term counts are measured independently; they are not advertised as vendor term-count equivalence.

The public native COBYLA optimizer uses a100-query cap and1e-4 tolerance. Its trajectory is not the unspecified hosted optimizer's trajectory. The best actually evaluated point, all100 query records, budget stop and convergence status are reported without repair. Source-style trimmed physical/WF-in-DFT estimates are kept separately from an additional actual **untrimmed** physical expectation. Trimming discards Hamiltonian terms, so the trimmed estimate is not a variational bound for the original Hamiltonian. The source's disabled diagonalization value remains explicitly historical.

Two nonzero thirty-angle programs are compared over every full sixteen-mode occupation amplitude, with native tapering lifted back to the original frame. Untrimmed native energies on Hartree-Fock and both coherent states are independently compared with spatial-integral PySCF CI contractions in the five-alpha/five-beta sector. Optional science packages remain outside production dependencies. No vendor SDK, credentials or cloud job is required.
