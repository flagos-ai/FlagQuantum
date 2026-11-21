# Verify both nozzle block encodings

All ten original code stages, the original 8 by 8 nozzle matrix with 21 entries, both Pauli and banded block encodings, fresh signed right-hand sides, and both unnormalized projections checked against the scaled matrix product.

From the repository root: `python -m examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.main`. Open the [Notebook](walkthrough.ipynb) for both projections.

The pinned CSR matrix is stored as plain numeric JSON. Pauli and diagonal decompositions are recomputed locally. Native synthesis prepares the data and selector amplitudes; the public Circuit composition applies Pauli SELECT or a bounded signed banded dilation, then unprepares the selector. These encodings have their own reported LCU scale and resources; no historical Gray-code gate count is claimed. Fixed local seeds make the generated right-hand sides reproducible. Absolute amplitudes are compared directly, with no sign correction or postselection renormalization.

[Provenance](provenance.json) and [source data license](LICENSE-data.txt).

The Notebook additionally executes the complete Pauli block as native qfunc:
fresh data preparation, selector PREPARE, controlled Pauli words and coefficient
phases, then inverse PREPARE through `within` / `apply`. The same 24-term
decomposition includes imaginary coefficients. It uses 12 total qubits,
including four clean work bits, and 430 logical instructions for the seeded
right-hand side; the original Circuit comparison retains eight qubits.
The native eight-qubit / zero-workspace request is refused, not replaced by
an opaque SELECT. These resource counts use different gate vocabularies and
are not a basis-level efficiency comparison. Every basis column and two
signed coherent right-hand sides match the complete independent Circuit state,
including absolute phase and all non-success selector branches. All four work
bits return exactly to zero. The banded encoding also has a complete native composition, described below.

The banded qfunc dilates the three real signed diagonals using existing
rotation tables, modular data-register increments/decrements and selector
PREPARE/unprepare. It retains the eight-row matrix, all 21 entries and LCU scale
3.8927056451476174. No SELECT response matrix is embedded in the native program.
All eight input columns and two signed coherent right-hand sides match the
complete six-qubit Circuit reference within 1e-12, without phase alignment.
The native route uses eight total qubits, two clean work bits and 104 logical
instructions for either source seed (depth 89); its six-qubit/zero-workspace
request refuses. The full Notebook still has ten original code stages.

Tightening `max_qubits` to seven selects a one-work-bit implementation with
156 logical instructions and depth 131. Both choices match every independent
input column and retain full flag-state amplitudes. The Notebook demonstrates
the same program under both constraints; it does not rewrite the quantum logic.
