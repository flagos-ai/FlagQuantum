# Two-qubit H2 UCCSD training across three encodings

Run the fixed two-orbital, two-electron H2 example:

```bash
python -m examples.algorithms.fermion_h2_tapered_uccsd --encoding parity
python -m examples.algorithms.fermion_h2_tapered_uccsd --encoding bravyi_kitaev
```

Jordan–Wigner remains the default. Each choice runs a two-qubit CPU statevector circuit with three float32 UCCSD angles
and complex128 simulation. From the fixed initialization it reports an
initial energy near -1.1347755088 Ha and reaches within 1e-6 Ha of
-1.1372701747 Ha after 160 Adam steps. The
[full-register example](FERMION_H2_UCCSD_ENCODINGS.md) uses the same Adam loop.

The sector is fixed by the -1 eigenvalue of each physical spin parity,
`(1-2n0)(1-2n2)` and `(1-2n1)(1-2n3)`. Each spin has two modes, so
the sector contains one alpha and one beta electron. The example encodes
the two parity operators with the Hamiltonian's encoding and checks that
each is a positive Z word before using the existing reducer. The reference
occupation is `1100` in physical order. It goes through the same encoder
before the reduction map selects the two free reference bits. Neither the
sector nor its reference is inferred.

Reuse the UCCSD factor order and the native excitation operator. Reduce the
Hamiltonian and each Hermitian excitation generator with the same sector
rules. The native reducer rejects a spin-changing excitation. Each factor
keeps the full-circuit convention `exp(i * theta * H / 2)` with
`H = i * excitation_operator(...)`; the order of the double's virtual
indices matters. The reduced terms commute within each factor. The example
checks this precondition, then reuses a single native Pauli product step with
weighted tensor coefficients. It is exact within each factor, and the
original order between factors is preserved.

Run `python -m pytest tests/integration/test_fermion_h2_tapered_uccsd.py -q`.
At two initial points and for each encoding, the tests compare the lifted
state with the matching four-qubit circuit from #1188 and check the lift
against an independent encoded occupation table. Lifted amplitudes and the
dense Hamiltonian restriction agree within 1e-12. Differentiable execution
is checked separately: the parity and BK energy and gradient comparisons
use an absolute tolerance of 1e-7 with the existing float32 angle path.
The tests also compare parameters and energy after one shared Adam update
and reject a spin-changing excitation in all three encodings. Lifting and
dense matrices are test-only; training evaluates native two-qubit
expectations.

This example covers only the fixed H2 fixture. It does not convert
arbitrary circuits or select sectors automatically. It adds no stable API,
dependency or performance claim and requires neither OpenFermion nor Cirq.
