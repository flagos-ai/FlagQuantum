# Verify both nozzle block encodings

All ten original code stages, the original 8 by 8 nozzle matrix with 21 entries, both Pauli and banded block encodings, fresh signed right-hand sides, and both unnormalized projections checked against the scaled matrix product.

From the repository root: `python -m examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.main`. Open the [Notebook](walkthrough.ipynb) for both projections.

The pinned CSR matrix is stored as plain numeric JSON. Pauli and diagonal decompositions are recomputed locally. Native synthesis prepares the data and selector amplitudes; the public Circuit composition applies Pauli SELECT or a bounded signed banded dilation, then unprepares the selector. These encodings have their own reported LCU scale and resources; no historical Gray-code gate count is claimed. Fixed local seeds make the generated right-hand sides reproducible. Absolute amplitudes are compared directly, with no sign correction or postselection renormalization.

[Provenance](provenance.json) and [source data license](LICENSE-data.txt).
