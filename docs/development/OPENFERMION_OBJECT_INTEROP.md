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
an installation hint. The separate `openfermion` extra pins the verified upstream
release; it is excluded from core, `all`, and `interop-all`. No registry, root
export or Stable Core change is introduced. Users may install
`flagquantum[openfermion]` in an interoperability environment; upstream may install
Cirq transitively.
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
must also fail. In the original EXT05 scope, MolecularData, InteractionOperator,
sparse matrices, circuits,
other upstream types, automatic environment installation and full compatibility
are outside scope.

## Acceptance

Check ordered fermion products against independent occupation-bit action,
including contractions and non-Hermitian complex terms. Check physical round
trips with real upstream classes, because upstream and native normal-order
conventions differ. Check small dense Pauli matrices, all native encodings through
an unchanged measurement consumer, identity/zero and invalid values. Import and
execute the native workflow in a fresh interpreter with OpenFermion/Cirq blocked.
Real-upstream tests carry the dedicated `openfermion` marker and run in the
isolated optional workflow, rather than being selected and silently skipped by
the native coverage lane. They skip when upstream is absent; fakes alone are not
evidence of upstream interoperability. Linux default tests remain CI's responsibility.

## Ownership and exit

Ecosystem maintainers own this explicit one-time migration aid. Allowed imports
are native algebra, standard library, and lazy upstream OpenFermion at the call
boundary. No native algorithm consumes foreign objects. Replacement uses the
same native operator consumers; removal is possible while this adapter remains
experimental, once old-object migration demand ends. No permanent shim, duplicate
contract or solver is added. The upstream Apache-2.0 dependency is separately
installed; this implementation copies no upstream source. The dependency review
and supply-chain scope are recorded in the dependency policy guide.


## InteractionOperator import extension

This separate extension adds only
`flagquantum.ecosystem.openfermion.from_openfermion_interaction(source)`.
The existing four conversions and native consumers retain their contracts. It
accepts an actual upstream `InteractionOperator` and returns the existing native
`FermionOperator` for

```text
constant + sum[p,q] one[p,q] a†p aq
         + sum[p,q,r,s] two[p,q,r,s] a†p a†q ar as
```

No extra factor of one half, integral permutation, Hermiticity assumption, or
coefficient tolerance is applied. Native ladder multiplication owns all CAR
signs, repeated-index zeros, and canonicalization. Non-Hermitian complex values
remain algebraic; native observable validation remains a separate operation.
This is not a conversion to `MolecularProblem`: spin-orbital operator
coefficients are not spatial chemist-notation integrals.

The mutable `n_body_tensors` mapping must contain exactly `()`, `(1, 0)` and
`(1, 1, 0, 0)`. One-body and two-body values must be dense NumPy arrays with
shapes `(n, n)` and `(n, n, n, n)`, including `n=0`, and integer, unsigned integer,
real floating or complex dtypes. Boolean, object and other dtypes fail. The
constant and every tensor entry must be finite numeric scalars, even if its
ladder product vanishes. Corrupted mappings, keys, shapes and scalar values
raise TypeError or ValueError. No input object or array is modified or retained.
Explicit register width is still supplied to native encoding.

Reading and validation take O(n^4) time with native storage proportional to the
nonzero canonical terms. The converter never constructs a dense Fock matrix.
It deliberately avoids upstream's convenience conversion because upstream
operator addition may prune tiny coefficients. NumPy is imported lazily only
at this explicitly selected tensor boundary and is already a dependency of the
separately installed upstream package. No production dependency, root export,
registry, solver, IR or Stable Core contract changes. Ecosystem maintainers own
this experimental one-way migration aid under the same removal condition above;
there is no reverse tensor export or full upstream compatibility claim. No
upstream source is copied; the optional upstream package remains Apache-2.0.

A runnable workflow in the separately installed interoperability environment:

```python
import numpy as np
import torch
import openfermion
import flagquantum as fq
from flagquantum.ecosystem.openfermion import from_openfermion_interaction
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.observables import to_observable

old = openfermion.InteractionOperator(
    0.25, np.diag([1.0, 2.0]), np.zeros((2, 2, 2, 2))
)
native = from_openfermion_interaction(old)
encoded = encode_fermion(native, n_modes=2, encoding="jordan_wigner")
value = fq.run(
    fq.Circuit(2).x(0), outputs=[fq.expectation(to_observable(encoded.operator))]
)
assert abs(float(torch.as_tensor(value.expectation()).reshape(-1)[0]) - 1.25) < 1e-6
```

Acceptance uses actual OpenFermion 1.7.1 objects, independent occupation-bit
matrices and upstream sparse operators, complex nonsymmetric tensors, quartic
order and normalization, numeric dtypes, zero-width inputs, tiny coefficients,
array ownership, malformed inputs, and all three existing native encodings.
The native fake boundary and fresh-process isolation run without upstream;
fakes do not replace the real-upstream tests. Run:

```bash
python -m pytest tests/unit/test_openfermion_native_boundary.py tests/integration/test_openfermion_interop.py -q
python -m pytest tests/unit/test_openfermion_conversion.py -m openfermion -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

The existing optional workflow runs real-upstream checks on Python 3.10 and
3.12. Linux broad test tiers remain CI's responsibility.
