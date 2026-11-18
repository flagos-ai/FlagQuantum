# LiH molecular chemistry and tapered VQE

Open [walkthrough.ipynb](walkthrough.ipynb) with the optional PySCF and
Matplotlib packages installed in the example environment, or run

```bash
python -m examples.synthesis.applications.chemistry.lih_chemistry.overview.main
```

The workflow regenerates the original neutral singlet LiH at 1.666944 Å in
STO-3G. It runs fresh RHF, MP2, CISD, CCSD and full FCI. Freezing the first
spatial orbital leaves five orbitals and one particle of each spin. Native
fermion algebra and Jordan–Wigner mapping produce the active Hamiltonian
from fresh integrals. An independent FlagQuantum integral build checks the
active FCI energy.

The workflow finds four independent diagonal Pauli symmetries and checks
each of the 16 sector spectra against a direct full-Hamiltonian restriction.
The native ordering interleaves alpha and beta modes. The notebook prints
the generators and Hartree–Fock sector in that ordering.

Ten retained UCCSD excitation generators form the six-qubit ansatz.
Independent Hermitian spectral evolution checks two nonzero parameter
vectors as complete complex states. The local COBYLA optimizer starts at
zero, evaluates the 231-term energy with exact local expectation, and stops
within 200 actual queries. The plot uses the measured local objective
trace and a fresh full-space FCI comparison.
