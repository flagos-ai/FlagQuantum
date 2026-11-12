# Explicit Pauli stabilizer reduction

## Stage contract

EXT06 extends the native fermionic science domain from diagonal Z parities to
arbitrary independent commuting signed Pauli words. This document was committed before implementation to freeze the
stage scope. The same feature PR now implements the interfaces below. It does not change a Stable Core export or schema.

`reduce_stabilizers(operator, *, n_qubits, generators, eigenvalues)` accepts a
native PauliOperator, an explicit register width, single-word native PauliOperator
generators with coefficient +1 or -1, and an integer eigenvalue +1 or -1 for each.
Generators must be nonidentity, independent over GF(2), pairwise commuting and
preserved by the operator. Complex operator coefficients are retained; observable
Hermiticity remains the existing measurement boundary's responsibility.

The result, StabilizerReduction, owns the reduced Pauli operator, a tuple of
existing native Instruction values describing a Clifford C in application order,
and an existing ReductionMap for the diagonal frame. C maps the requested joint
sector to fixed pivot Z eigenvalues. For the diagonal embedding J, the original
sector embedding is E = C dagger J and H E = E H_reduced. The old affine map alone
cannot represent an X/Y sector because one reduced basis state may lift to a
superposition. This is why the result must carry the Clifford as well as the
existing diagonal map. No second circuit IR, registry or backend is introduced.

`StabilizerReduction.reduce_operator` applies that same map to another preserving
operator. It rejects nonpreserving operators rather than returning a silently
projected zero. `stabilizer_clifford` lowers C (or C dagger) to an ordinary Circuit;
`lift_stabilizer_state` uses the existing bounded affine embedding and the native
inverse Clifford to restore a state in the original register. The symbolic owner
never imports Circuit, runtime, simulators or third-party scientific packages.

## Conventions and limits

Qubit zero is the most significant statevector bit. Free qubits retain ascending
original qubit order in the diagonal Clifford frame. Generator signs and the
user's eigenvalues both affect the selected sector. An empty generator list is
the identity reduction. Complete constraints may leave zero qubits, represented
by a scalar Pauli operator and a one-element reduced vector. No zero-qubit
Circuit is required for state lifting.

Symbolic reduction never constructs dense matrices or statevectors. State lifting
is a separately invoked CPU complex128 inspection path, with the existing
20-qubit limit applied before allocation. There is no gradient, accelerator,
distributed, automatic symmetry discovery or arbitrary tapered-ansatz claim.
The user supplies the sector; minimizing all sectors is not an implicit solver.

## Acceptance

- Check H, S dagger and CX Pauli conjugation against independent dense matrices,
  including Y signs and multi-qubit words.
- Check every supplied generator eigenvalue and the complete sector identity
  H E = E H_reduced, not only the minimum eigenvalue.
- Compare direct projector spectra and expectations for independent signed X/Y/Z
  generators, random small Clifford-rotated systems and all sectors.
- Preserve the existing diagonal API and demonstrate that general reduction and
  its diagonal implementation give equivalent physical sector spectra.
- Run a native molecular example, with exact small-sector reference checks and
  a plain measured reduced-state energy. Add it to the executable example index.
- Reject dependent, anticommuting, out-of-register, identity, nonunit-coefficient
  and malformed generators, invalid sectors and nonpreserving operators.
- Check state validation, preallocation budgets, zero-qubit sectors, round-trip
  circuit semantics and execution with OpenFermion/Cirq imports blocked.

The stage uses independent algebraic references and native execution. No upstream
source or new dependency is required. Linux full suites and version matrices run
in CI; each implementation commit includes focused local evidence.

## Executable workflow

Run `python -m examples.algorithms.fermion_tapering`. The fixed H2 example
uses a signed X/Y symmetry in a rotated qubit representation, reduces four
qubits to two, measures -1.1372701747 Ha and particle number 2, and lifts the
state with a full-register eigenvector residual below 1e-12. The two-qubit state
comes from small dense diagonalization and is explicitly synthesized through
the existing state-preparation algorithm before fq.run; it is not a scalable
solver. Example values are asserted by tests/test_algorithm_examples.py.

Run `python -m pytest tests/unit/fermion/test_stabilizer.py -q` for mathematical
and boundary checks, and tests/integration/test_native_fermion_isolation.py for
execution with external scientific SDK imports blocked.
