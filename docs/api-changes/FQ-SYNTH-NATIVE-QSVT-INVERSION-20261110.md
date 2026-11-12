# Native QSVT matrix-inversion composition

Status: accepted locally with independent full-state evidence and fresh corpus replay.

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

## Accepted evidence and bounded inverse execution

All ten native acceptance cases passed: the unchanged forward/original and two
perturbed phase cases in the first run, and the seven affected inverse, budget
and phase-length cases after correcting the inverse test boundary. Ninety
related tests and 29 catalog/API cases passed, with strict typing of five files
and all local quality checks. Fresh replay preserves all 216 original hashes,
semantic outcomes and resources; six approved revisions execute. All 29 pinned
production files remain unchanged.

A combined forward-plus-inverse exceeds the existing 100,000-operation internal
construction cap. Increasing a user resource constraint does not raise that
cap. Retained failure logs establish this distinction. The final checked inverse
is compiled separately within the cap and executed through Circuit on the
independent full forward state, restoring the input and every clean work bit.
Neither a compiler limit nor a refusal was weakened to pass acceptance.

Native resources are five physical qubits, one clean work bit, 68,023 operations
and depth 59,977. The original uses four physical qubits, no work, 67,915
operations and depth 59,872. Both absolute amplitude errors are below 1.2e-12
with zero work leakage. Native accepted coverage is now 196/216 (90.7%), with
20 pending; Notebook coverage remains 24/213 and merged revision-nine scopes
remain 14/19. Immutable evidence is recorded under `up06-native-qsvt-inversion`.
