# Periodic Laplacian block encodings

```bash
python -m examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.overview.main
```

The complete Notebook retains all seven source code stages: a 32-point periodic
one-dimensional grid, the two-dimensional grid `(2, 4)`, both classical matrix
constructions, native quantum programs, actual full-state execution, projected
blocks and fidelity checks. Both source programs use the zero input. The width
limits remain seven and six, with the original depth objective and zero compiler
workspace. No grid is resized.

The native entry reuses the existing Fourier-shift quantum functions. Users see
named numeric registers and projected grid amplitudes. The split-coordinate
implementation represents the packed coordinate as `j = 2*j1 + j0`; coordinate
zero is the least significant axis, matching the source reversed Kronecker order.
The two-dimensional scale is one half of the sum of the one-dimensional blocks.

The source display removes the first amplitude's global phase and prints its
real part. That display is retained separately. Acceptance compares the complete
complex state with independent permutation-matrix Circuits and compares the raw,
unnormalized projected block with the fresh finite-difference matrix. Normalized
fidelity cannot conceal a sign, scale or phase discrepancy. Projection probability
is reported without replacing the full state with a postselected result.

`model.py` owns the bounded classical matrices; `workflow.py` executes the native
programs and compares both references; `main.py` and `walkthrough.ipynb` are thin
user entries. Circuit drawings replace the source platform viewer. No cloud job,
source importer, new API or dependency is needed. This is a local correctness
example, without a hardware, speedup or scalability claim.
