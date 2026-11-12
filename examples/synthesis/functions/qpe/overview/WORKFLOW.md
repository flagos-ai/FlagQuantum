# Qpe workflow acceptance

Retain both source stages: CRZ(pi) on |11> and the two-bit fractional counter; then the complete commuting IZ/ZI/II Hamiltonian with coefficients -0.125/-0.25/0.375. A qpe callback with one Suzuki step exactly replaces the original power callback because all terms commute; retain identity phase. Use exact typed phase semantics for CRZ. Original shot count is unspecified; label 1000 local demonstration samples per stage.

Every original code cell is reviewed and hash-bound in provenance. Use native
Python qfunc definitions, synthesis, local CPU execution and named readout.
Seeded host samples are drawn from exact local probabilities; this does not
claim external provider sampling. Resource summaries replace the source graphical
program display. No source-language loader or remote credentials are needed.

Independent Circuit references check complete complex amplitudes, absolute phase
and named bit order. Dropped data remains in the joint state, inaccessible to
user code. A zero workspace report must not be interpreted as a reset guarantee
for that environment. Hadamard and SWAP references retain those data explicitly;
QPE references construct independent dense matrix powers and inverse Fourier.

Notebook acceptance is separate from frozen quantum-program acceptance. This is
local CPU evidence, with no hardware, distributed or resource-optimality claim.

```sh
python -m pytest tests/integration/test_synthesis_interference_notebooks.py -q
```
