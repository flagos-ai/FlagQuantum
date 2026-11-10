# Native rotation tables and reversible matrix-QPE composition

The continuing native corpus migration admits `select_rotation(axis, angles,
selector, target)` with literal axis X/Y/Z, a finite static real table of
2**N entries and a one-to-eight-bit initialized selector. Selector bits are
raw storage words even when signed/fractional metadata is present. Position
zero is the least significant selector bit. Whole registers and existing static
array views are admitted. The one writable target qubit must be physically
disjoint from all distinct selector locations. Existing initialization,Const,
view and resource checks remain authoritative. Host invocation raises TypeError.

Compiler owns the existing Walsh/Gray table expansion formerly implemented in
the Qmod adapter. Extraction must preserve that exact gate sequence, including
its closing CX edge and conditional 2pi signs. Ecosystem remains responsible
for source parsing and calls the shared Compiler transformation; native capture
uses that same transformation. No second numerical algorithm,physical opcode,
executor,public record type or root export is added. Pure within/adjoint regions
and plain mutable helpers retain existing borrowing checks; general coherent
control of the entire table remains outside this admission.

Matrix-only QPE may be captured in root pure unitary regions, including within
preparation and adjoint. The allocated/unused all-fractional counter contract
stays unchanged. No allocation or publication occurs inside these regions;
callers allocate visible indicator outputs before reversible composition.
Existing inverse_effects owns inverse matrix powers and Fourier rules. Nested
helper ownership and arbitrary/flexible callbacks remain refused.

Acceptance covers every table entry,nonzero target columns,complex/coherent
inputs,raw signed/fractional indexing,view bit order,conditional global phases,
all three axes,inverse composition,alias/initialization/width/table failures and
actual budgets. Matrix HHL originals require complete independent Circuit
vectors with original matrices,tables,widths and preparations,and two nonzero
parameter variations. Non-normalized original inputs remain refused with an
explicit approved normalization path. Fresh216replay must preserve source
identities,outcomes and resources after shared extraction. This admits local
implementation only,not release or algorithmic performance claims.
