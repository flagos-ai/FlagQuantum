# LiH energy from molecular phase estimation

Run the [notebook](walkthrough.ipynb) or the equivalent local workflow from the
repository root.

```sh
python -m examples.synthesis.applications.chemistry.qpe_for_molecules.overview.main
```

The selected system is neutral LiH at 1.596 Å in STO-3G. FlagQuantum regenerates
the integrals and solves RHF and full CI. The QPE model freezes one core orbital,
keeps five active spatial orbitals and two electrons, then encodes the 64
occupation words in the Hartree–Fock parity sector. Its classical coordinate
change uses the source encoding. The stored example supplies the Pauli product
order and a regression reference. Every coefficient used in synthesis comes
from the fresh molecular calculation.

The workflow applies the original 0.03 Ha cutoff and L₁ normalization, evaluates
the ordered double-commutator bound, and compares the two theoretical repetition
schedules with the experimental schedule `1, 1, 1, 2, 4, 10, 26`. The seven
controlled evolutions each use their own second-order product formula. The
native program combines only adjacent commuting factors to fit the effect budget.
The inverse QFT produces a signed fractional phase on seven bits.

The local CPU workflow draws 1,000 samples with a fixed seed from the native
phase marginal. It reports the mode and applies the source's minimum peak height
to measured bins in energy order. The full and active CI energies and the phase
bin width are reported separately. The sampling seed is a local reproducibility
choice. The sample, finite phase grid, Hamiltonian truncation and product formula
all affect the QPE estimates.

The example runs locally with FlagQuantum and Matplotlib. The native solver supplies the RHF and CI references.
