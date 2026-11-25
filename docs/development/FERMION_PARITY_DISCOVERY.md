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


## Explicit-sector training composition

```bash
python -m examples.algorithms.fermion_parity_training
python -m pytest tests/integration/test_fermion_parity_training.py -q
```

The separate example uses H = n0 + n1 - 1 - 0.7*(c0† c1† + c1 c0).
Total parity is conserved; particle number is not. The explicitly selected even
sector spans vacuum and double occupancy. In that occupation basis its matrix
is [[-1, -0.7], [-0.7, 1]], with ground energy -sqrt(1.49). The odd sector has
zero energy. No sector search or automatic ground-sector selection is performed.

Discovery and reduction happen once with static native coefficients. A native
one-qubit RY preparation and `fq.run` expectation provide the trainable objective;
PyTorch owns differentiation and optimization. Starting at angle 1.7, 200 Adam
steps with learning rate 0.05 lower energy from -0.5653208730 to -1.2206555611.
The gradient agrees with central differences below 1e-9. Dense lifting checks
energy agreement and zero odd-sector leakage. The final number expectation is
approximately 0.1808, rather than a fixed integer. Dense inspection is confined
to the two-mode example and tests; no dense solver initializes the training.

Integration tests substitute all three encodings at the existing encoded
operator boundary. Physical parity is encoded before discovery, so encoded
qubit indices are never assumed to be occupation-mode labels. Independent
occupation-space matrices verify both coupling signs, energies, gradients and
optimizer updates, and opposite-sector lifting checks the requested constraint.
A fresh interpreter executes training with OpenFermion, Cirq, PySCF, SciPy and
h5py blocked. PyTorch optimization may use NumPy; NumPy-free training is not
claimed. Vector angles fail before circuit construction.

This is an experimental `single_device_fast_path` composition of existing
interfaces. Symbolic discovery/reduction remain nondifferentiable static
operations; their capability registration is unchanged. Only the reduced circuit
angle is trained. No general ansatz transformation, coefficient/SCF gradient,
GPU/distributed qualification, performance or general convergence claim follows.
