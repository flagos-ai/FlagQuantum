# Diagonal Z2 symmetry discovery

## Scope and ownership

Algorithms discovers a basis of encoded Z-only Pauli words commuting with a
native `EncodedOperator`. `discover_z_parities(encoded, *, candidates=None)` in
`algorithms.fermion.symmetry` returns tuples of encoded qubit indices, directly
accepted by the existing `reduce_z_parities`. With no candidates, it searches
all diagonal words. With explicit independent candidate words, it searches
exactly their binary product span, including commuting products of candidates
that individually fail to commute. It returns an independent deterministic
basis, not every symmetry. Empty candidates search only the identity and return
an empty basis. Identity generators are omitted.

For each Pauli term, its X/Y support gives one GF(2) commutation equation. A
symbolic binary nullspace solves these equations without dense state allocation.
Exact canonical coefficients determine which terms exist; there is no numerical
threshold. Complex coefficients do not require Hermiticity for discovery.
Malformed candidates, duplicates/dependencies and out-of-register indices fail
before discovery. Input and output words refer to encoded qubits, not occupation
modes; JW/parity/BK conventions stay owned by `encoding.py`.

Discovery does not select eigenvalues. An explicit particle count N determines
only total parity (-1)**N, not a unique N-particle subspace. Separate spin
populations or other explicit parity constraints can select more sectors, but
must not be inferred from a count alone. A reference occupation is valid for
choosing a joint parity sector only when that choice is explicitly intended;
it is not evidence of number conservation or of the physical ground sector.
General X/Y-generator discovery, sector minimization, matched tapered ansatzes,
coefficient gradients and accelerator/distributed claims remain outside scope.
No Stable Core export, schema, dependency or new result type is introduced.

## Acceptance and contributor path

The implementation must prove completeness against exhaustive small-register
commutators, including restricted candidate spans, identity/zero operators and
commuting products of noncommuting candidates. Verify independent output,
determinism under term/candidate reordering and each declared encoding. For
selected sectors, check the complete intertwining identity H E = E H_reduced
and eigenvalues of all lifted basis columns, not only ground energies. Include
a number-nonconserving pairing model that still has total parity, and a
parity-breaking source that has none. Native examples must run with external
quantum/chemistry SDK imports blocked.

Use the existing reduced-operator/state-embedding boundary unchanged. The
example compares a discovered total parity with an explicitly supplied total
parity while keeping its reduction and measurement consumers unchanged. This
is a replacement check, not a second reduction implementation.


## Executable workflow

```bash
python -m examples.algorithms.fermion_parity_discovery
python -m pytest tests/unit/fermion/test_parity_discovery.py tests/integration/test_fermion_parity_discovery.py -q
```

The two-site JW Hubbard model at t=1, U=4 has two independent diagonal
symmetries. Searching only the total-parity candidate (0,1,2,3) returns that word.
N=2 explicitly selects eigenvalue +1, reducing four qubits to three. The even
sector includes N=0,2,4; at these coefficients its ground energy is
-0.8284271247. The example prepares the small reduced eigenstate, measures with
`fq.run` and verifies its lifted eigenvector residual below 1e-12. This is
bounded CPU `single_device_fast_path` inspection; no training, capacity or
scalability claim follows. The example does not choose the two unrestricted
generators' eigenvalues from N alone. Encoding-replacement tests independently
construct physical total parity and verify the same reduction consumer through
JW, parity and BK, including the presence of all three even particle counts.
