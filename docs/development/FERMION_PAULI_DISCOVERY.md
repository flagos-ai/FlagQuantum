# Commuting Pauli symmetry discovery

## Scope and ownership

Algorithms extends the existing symbolic stabilizer boundary with
`discover_pauli_symmetries(operator, *, n_qubits, candidates=None)` in
`algorithms.fermion.stabilizer`. It accepts native Pauli values and returns a
tuple of native single-word Pauli operators, directly consumed by existing
`reduce_stabilizers`. No Stable Core export, dependency, result type, registry,
IR, runtime change or dense allocation is needed for discovery.

The discovered words have coefficient +1, are independent over GF(2), commute
pairwise and commute with the canonical input operator. The complete Pauli
centralizer can contain anticommuting pairs. The result is a deterministic
maximum-size commuting subset, not a basis of that entire centralizer and not
every conserved observable. Degenerate systems can admit several equally large
choices. The fixed binary ordering selects one; no smallest-support, chemically
preferred, symmetry-adapted ansatz or optimal measurement claim follows.

An explicit register width retains idle qubits, including zero-width registers.
Unrestricted search uses all X/Z binary labels. Independent candidate words
restrict search to their binary span modulo scalar phases; an empty list searches
only identity and returns no generator. Candidates may anticommute or fail to
commute with the input individually. Their products can still produce valid
constraints. Candidate signs are ignored, and returned representatives are
always positive Hermitian words, so caller-supplied signed eigenvalues cannot be
transferred without adjusting their signs. Identity, multiword, nonunit,
dependent, out-of-register and malformed candidates fail before discovery.

The algorithm solves termwise binary symplectic commutation equations using the
existing domain GF(2) elimination. Canonical rows make equivalent input spans
deterministic. Symplectic elimination retains the radical and one member of each
anticommuting pair, giving (d+r)/2 generators for centralizer dimension d and
radical dimension r within the searched span. Canonical row reduction then fixes
the output representation. Exact canonical nonzero coefficients define the
constraints; there is no tolerance. Complex non-Hermitian input is permitted;
measurement Hermiticity remains the observable boundary's responsibility.

## Sector and failure boundaries

Discovery selects no eigenvalues, particle number or ground sector. Indices
refer to encoded qubits, never inferred fermion-mode labels. Replacing an
explicit generator with a negative representative requires changing its sector
eigenvalue too. Every requested sector remains a separate explicit reduction.
Existing Z-only discovery retains its signature, candidate-span meaning and
canonical results; unrestricted general discovery need not select that same
commuting subgroup. Restrict candidates to the desired physical constraints
when their identity matters.

This is an experimental symbolic CPU path. State lifting and dense comparisons
are separate bounded inspection tools. Circuit training, automatic transformed
ansatzes, coefficient/SCF gradients, accelerator/distributed qualification,
performance and scalability claims remain outside scope. Capability registration
belongs to a separate Integration-owned change.

## Acceptance before delivery

- Enumerate small-register Pauli matrices independently and check exact
  centralizer membership, independence, pairwise commutation and maximality.
  Every centralizer word commuting with the result must belong to its span.
- Cover genuinely anticommuting centralizers, restricted spans, commuting
  products of individually invalid candidates, signed representatives and
  equivalent candidate bases. Check term/candidate order determinism.
- Preserve exact tiny terms, cancellation, zero/identity and idle/zero-width
  semantics. Verify wide symbolic registers without dense state allocation.
- Substitute discovered generators for supplied constraints while leaving
  reduction, measurement and lifting consumers unchanged. Check all joint
  sectors, full intertwining H E = E H_reduced, lifted generator eigenvalues
  and dense projector spectra. Include X/Y words and JW/parity/BK replacement.
- Reject malformed native inputs and candidate spans. Execute a documented
  native example with external quantum/chemistry libraries blocked.
- Run focused local scientific and API checks; full Linux suites run in CI.

## Contributor path

The implementation and focused tests will be indexed here with measured results
before the feature PR is opened. Symbolic discovery owns only native algebra and
binary constraints; lowering owns the existing Clifford/state embedding and
`fq.run` owns execution. The existing reduction entry point is the replacement
boundary, so consumers do not learn discovery internals.
