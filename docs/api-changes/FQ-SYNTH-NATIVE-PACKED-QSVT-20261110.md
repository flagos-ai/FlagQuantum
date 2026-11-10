# Native packed-register QSVT compositions

Status: both independent native/Circuit probes passed; formal acceptance pending.

The Classiq-paper QSVT and heat-equation QSVT share modular packed-word shifts,
static unitary composition and block-zero projected rotations. Preserve their
six-bit and three-bit data sizes, complete phase arrays and block ancillas.
Expose the source's packed data/deletion word as an explicit native output;
name and document the output permutation rather than pretending its registers
match the source struct. No data or nonzero auxiliary amplitude may be dropped.

Existing raw-word increment, checked adjoint, inplace preparation and polynomial
phase capture suffice. In the root program, a within/apply arithmetic marker
extracts the deletion or block-zero predicate and is cleaned immediately after
its phase use. Projected RZ is represented as two exact phase polynomials,
including its constant phase. Mutable helpers remain pure effects: do not relax
borrowed ownership rules, the eight-bit control predicate cap, or the 32-solution
sparse-phase cap to accept these programs.

The heat source passes four phases but its loop consumes only the first three.
Preserve all four values and that actual execution behavior; test explicitly
that perturbing the unused fourth phase has no effect. This migration does not
silently repair upstream algorithm intent. Initial heat amplitudes are embedded
in the deletion-zero half of its four-bit packed word, retaining every original
data amplitude and zero initial deletion bit.

Acceptance uses independent Circuit block matrices, modular permutation and
analytic projected-phase diagonals, not native phase-polynomial lowering.
Compare all absolute complex amplitudes, every work amplitude and explicit
source/native port mappings for both unchanged originals and two nonzero phase
variants. Test heat nonzero input variants, phase-array lengths, budgets and
checked inverse of the borrowed blocks. Report resource overhead honestly.

Complete final216 plus approved6 replay, immutable evidence and catalog/ledger
updates before credit. Native coverage starts at197/216; Notebook coverage
remains24/213. Classification is `single_device_fast_path`. No integration PR,
push or full remote CI before all216 native programs pass acceptance.
