# Optional OpenFermion operator conversion

## EXT05 contract

This demand-driven stage converts existing OpenFermion FermionOperator and
QubitOperator values at the ecosystem boundary. It does not restore an external
execution path. The sole native representations remain the existing
`flagquantum.observables.fermion.FermionOperator` and
`flagquantum.core.algebra.pauli.PauliOperator`.

The concrete `flagquantum.ecosystem.openfermion` module exposes
`from_openfermion_fermion`, `to_openfermion_fermion`,
`from_openfermion_qubit` and `to_openfermion_qubit`. Native imports, including
importing this adapter module, must not import OpenFermion or Cirq. Each explicit
conversion lazily checks for OpenFermion; absent packages raise ImportError with
an installation hint. No dependency, optional extra, registry, root export or
Stable Core change is introduced. Users install upstream OpenFermion separately
in an interoperability environment; upstream may install Cirq transitively.
The production native environment requires neither library.

Fermion import composes the ordered ladder products with native create and
annihilate operations, retaining contractions and signs without a second
normal-ordering implementation. Export preserves the native monomial's operator
order, even if upstream normal-ordering uses another convention. Qubit import
preserves Pauli factors and complex coefficients. Identity constants, exact
zeros and non-Hermitian operators are allowed. No tolerance discards small terms.
Labels retain their meaning; neither algebra stores idle register width, so
callers supply explicit n_modes or n_qubits at encoding and inspection boundaries.
Input objects are not modified and their mutable terms are never retained.

Malformed type, mapping, factor shape, repeated Pauli index, axis or ladder
action raises TypeError or ValueError at conversion. Coefficients must be finite
numeric scalars; bool, symbolic values and tensors are rejected. Modes must be
nonnegative integers. External instances whose mutable terms have been corrupted
must also fail. MolecularData, InteractionOperator, sparse matrices, circuits,
other upstream types, automatic environment installation and full compatibility
are outside scope.

## Acceptance

Check ordered fermion products against independent occupation-bit action,
including contractions and non-Hermitian complex terms. Check physical round
trips with real upstream classes, because upstream and native normal-order
conventions differ. Check small dense Pauli matrices, all native encodings through
an unchanged measurement consumer, identity/zero and invalid values. Import and
execute the native workflow in a fresh interpreter with OpenFermion/Cirq blocked.
Optional tests skip when upstream is absent; fakes alone are not evidence of
upstream interoperability. Linux default tests remain CI's responsibility.

## Ownership and exit

Ecosystem maintainers own this explicit one-time migration aid. Allowed imports
are native algebra, standard library, and lazy upstream OpenFermion at the call
boundary. No native algorithm consumes foreign objects. Replacement uses the
same native operator consumers; removal is possible while this adapter remains
experimental, once old-object migration demand ends. No permanent shim, duplicate
contract or solver is added. The upstream Apache-2.0 dependency is separately
installed; this implementation copies no upstream source.
