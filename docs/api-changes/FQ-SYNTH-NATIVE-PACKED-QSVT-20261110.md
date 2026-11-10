# Native packed-register QSVT compositions

Status: accepted locally with independent complete-state and fresh replay evidence.

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

## Accepted evidence

All 23 new native cases and related regressions passed together (155 tests).
Strict typing covers seven files; Black, Ruff, 29 catalog/API tests, catalog,
source-of-truth, architecture, maturity, links and classification pass. The
fresh 216-source replay preserves every prior source hash, semantic outcome
and resource record. All six approved revisions execute and all 29 production
pins remain unchanged.

The paper QSVT uses 15 physical qubits including five clean work bits, 17,663
operations and depth 14,619; the original uses 14 qubits with four work bits,
2,798 operations and depth 2,270. Heat QSVT uses 11 qubits with three work bits,
210 operations and depth 160; its original uses 11 qubits with three work bits,
267 operations and depth 202. Native absolute errors are below 3.2e-13 and
1.5e-16; original errors are below 4.5e-14 and 1.1e-15. All work leakage is zero.
The paper cost gap is retained explicitly rather than hidden as an optimization.

Native accepted coverage is 199/216 (92.1%), with 17 pending. Notebook coverage
remains 24/213 and revision-nine merged scopes remain 14/19. Immutable evidence
is recorded under `up06-native-packed-qsvt` in the delivery ledger.
