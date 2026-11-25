# Structured matrix block encodings

All ten source code stages and the original checkerboard, Toeplitz, symmetric tridiagonal, and 4 by 4 grid Laplacian models. Preserve the dimensions 8, 16, 16, and 16, the respective scales 8, 4, 3, and 5, and the original uniform input.

Run from the repository root with `python -m examples.synthesis.community.paper_implementation_project.block_encoding.select_structures_be.overview.main`. The [Notebook](walkthrough.ipynb) keeps the signed matrix-product comparisons visible.

Generate signed coefficients from a reproducible local seed. Use native synth for amplitudes and a bounded public Circuit SELECT for each structured address map. Compare absolute complex amplitudes directly to each declared matrix product. The native Laplacian encodes the same matrix scaled by the magnitude of its diagonal. The historical checkerboard and Laplacian prints show opposite overall signs between their projected and theoretical vectors. This workflow preserves the signed theoretical matrices and verifies their native projections directly.

[Source provenance](provenance.json).
