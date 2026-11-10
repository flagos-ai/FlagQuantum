# Native QSVT matrix-inversion composition

Status: dependency review complete; native probe and independent acceptance pending.

The frozen matrix-inversion program has a two-bit state, one block bit, one
signal auxiliary, an eight-by-eight real block encoding and 70 inversion phases.
Express the complete program with existing native allocation, real amplitude
preparation, unitary application, adjoint, bounded static loops, checked control,
X and RZ. No new public QSVT marker or model-specific Compiler rule is required.
Keep all original numerical literals. Flatten the source struct into explicitly
named native state and block ports and record their mapping; do not silently
change matrix endianness or discard the auxiliary.

The inversion sequence begins with H, applies the first projected RZ before
any block, alternates U-dagger then U, and finishes with the final projected
rotation and H. Preserve absolute phases, including the large final angle.
The independent Circuit must construct projector-conjugated phase diagonals
analytically, rather than replaying the native controlled-X construction. Compare
all 16 logical amplitudes and every clean work amplitude, including variants
with two nonzero phase perturbations. Verify the original literals independently,
the postselected scaled linear solution for the original phases, inverse
composition, port order and resource refusals. Do not claim phase-sequence
synthesis, general matrix inversion, or complete classical Notebook workflows.

Run the final full 216-source and approved-six replay, archive evidence, and
update catalog and delivery ledger only after independent acceptance. Existing
source resource baselines remain immutable. Path classification is
`single_device_fast_path`. Starting native coverage is 195/216; Notebook
coverage is separately 24/213. No integration PR or full remote CI before
216-program native acceptance.
