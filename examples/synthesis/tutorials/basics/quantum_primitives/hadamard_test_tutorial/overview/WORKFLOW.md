# Hadamard Tutorial workflow acceptance

Retain four data qubits, helper composition with within/apply, controlled QFT and the original 100000 shots. Only the ancilla is public; drop retains an entangled environment. Independent validation checks the entire joint state and reduced purity, not zero hidden work.

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
