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
transferred without adjusting their signs. The returned basis can also contain
products of candidates, so an old eigenvalue list cannot simply be reused.
Identity, multiword, nonunit,
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

Run the shortest workflow:

```bash
python -m examples.algorithms.fermion_pauli_discovery
python -m pytest tests/unit/fermion/test_pauli_discovery.py tests/integration/test_fermion_pauli_discovery.py -q
```

The three-qubit encoded model is H = Z2 + 0.4*Z1*X2 + 0.2*Y0 + 0.3*X1*Z2.
Discovery returns positive Y0 and X1*Z2. The example explicitly chooses their
+1 and -1 eigenvalues, reducing three qubits to one. In that selected sector the
two energies are -0.1 +/- sqrt(1.16); native measurement of the small reduced
eigenstate gives -1.1770329614. Its lifted eigenvector residual is below 1e-12.
The other three joint sectors are checked separately by independent projectors
and analytic spectra; no global ground-sector claim follows from this example.

Change a static coefficient and run the focused tests. Exhaustive one-, two-
and seeded three-qubit matrices verify centralizer membership, maximum commuting
rank and completeness within the selected subgroup. Candidate-span tests cover
anticommuting factors, binary products, equivalent bases and discarded signs.
A 128-qubit case blocks dense tensor constructors. Integration replaces supplied
X/Y constraints, checks complete embeddings and measurements, and substitutes
JW/parity/BK physical spin-parity suppliers. The latter deliberately selects the
joint sector of an explicitly intended reference occupation, rather than
inferring spin sectors from total particle number. The existing Z-only result is
preserved when general discovery is restricted to the full diagonal span.
A fresh interpreter runs the example with OpenFermion, Cirq, PySCF, SciPy, h5py
and NumPy blocked.

Symbolic discovery owns only native algebra and binary constraints; lowering
owns the existing Clifford/state embedding and `fq.run` owns execution. Both
Z-only and general discovery reuse one private GF(2) nullspace implementation.
The existing reduction entry point is the replacement boundary, so consumers
do not learn discovery internals. Discovery does not construct a circuit or
choose a state. Only the explicitly invoked small example diagonalizes a matrix.
