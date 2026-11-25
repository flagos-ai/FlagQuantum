# Joint group/time Lanchester HHL

The source group record becomes two named native outputs: `rabbits` (one bit)
and `time_index` (unsigned four-bit QNum). They retain independent readout and
the source physical order. The explicit `(rabbits, time_index)` sequence lists
least-significant storage first. It jointly prepares all 32 original amplitudes
and targets the unchanged 32x32 matrix, preserving entanglement across fields.
Preparing the fields separately would change the state.

Allocate both fields, then call `inplace_prepare_amplitudes` on their explicit
sequence. This applies the existing reversible preparation unitary on already
initialized storage; it does not allocate, reset or assert zero. QPE in `within`
and the unchanged 64-entry Y rotation table in `apply` reuse Compiler lowering.
The matrix decomposition needs four clean work bits. Original and native costs
are equal: 16 physical bits, 20213 operations and depth 18016.

The independent Circuit uses Householder preparation, direct controlled matrix
powers, dense Fourier and direct table blocks. Acceptance compares all complex
amplitudes without global-phase alignment at the original vector and two nonzero
variations, every named field/phase marginal and clean workspace. Exact
`numeric_probabilities` keys retain unsigned time and six-bit Fraction phases.
Width, work, operation, depth and memory limits reject infeasible requests.

Joint sequences are flat, static and explicitly ordered. Existing array views
may be elements; each contributes least-significant-first locations. Expanded
locations must be distinct, initialized and writable. Nested/overlapping/Const
or uninitialized selections refuse. Vector bounds stay 2..256 and matrix sizes
2..32. Direct inplace subviews remain refused; use an explicit sequence to make
partial/joint selection visible. No general qstruct class or implicit record
reinterpretation is added. Complete Notebook workflow remains pending.
