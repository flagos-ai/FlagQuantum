# Nozzle linear systems with Chebyshev LCU

Run the [notebook](walkthrough.ipynb) or the local workflow from the repository
root.

```sh
python -m examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.main
```

The eight-row case uses the original 24-to-20 Pauli trim and degree-31 inverse.
The sixteen-row case uses a banded block encoding and degree 255, once with
exact coefficient preparation and once with the original 0.03 preparation
budget. The examples load both original right-hand sides and retain every
data row, selector bit, and signed odd coefficient. Matrix data and right-hand
sides follow the [upstream license](../../verify_block_encoding/overview/LICENSE-data.txt).

The workflow synthesizes the real input, block weights and inverse LCU
coefficients with native qfuncs. It constructs the Pauli and banded selectors
from the matrix and verifies each encoded top-left block. Each binary control
then applies an exact matrix power of the corresponding walk over the local
data and block registers. The large case has seven controls for powers from
1 to 64. The resource report counts these represented matrix instructions in
the simulator circuit. Gate-level costs require a separate target decomposition.

The zero-selector amplitudes recover the signed inverse action. An independent
sequential walk checks all amplitudes, and a singular-value polynomial checks
the projected result. An exact solve of each original matrix provides the
reference for the final solution plots and L₂ errors.
