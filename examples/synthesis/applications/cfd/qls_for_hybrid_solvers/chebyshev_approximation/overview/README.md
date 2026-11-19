# Chebyshev approximation for the nozzle inverse

Run the [notebook](walkthrough.ipynb) or the equivalent CPU workflow from the
repository root.

```sh
python -m examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.main
```

The example reconstructs the original 16 by 16 nozzle matrix from its 45 CSR
entries and computes the singular values after banded block scaling. It compares
four fits at the original degrees: relative-error minimax at 127 and 255,
uniform-error fitting at 255, and CKS truncated from degree 2575 to 255.
The relative fit uses the analytic shifted Chebyshev residual. The uniform fit
uses 160 Lawson reweighting steps, and the error report measures both fits on
the source singular-value interval. The original matrix is distributed under
the [upstream data license](../../verify_block_encoding/overview/LICENSE-data.txt).

For the degree-255 relative fit, the example reproduces the source endpoint
line fit and turns all 128 alternating coefficients into seven-bit LCU weights.
The approximate route averages equal-size adjacent probability groups with an
explicit 0.03 amplitude L₂ bound, checking the bound against the executed
state. The native qfunc prepares group weights on the high bits and applies
Hadamards to the low bits. The exact route prepares all 128 weights directly.
Both routes count actual RY and CX instructions, and use the observed
probabilities to reconstruct the signed coefficients and resulting polynomial.
