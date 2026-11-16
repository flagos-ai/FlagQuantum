# Two-qubit JW H2 UCCSD training

Run the fixed two-orbital, two-electron H2 example:

```bash
python -m examples.algorithms.fermion_h2_tapered_uccsd
```

It runs a two-qubit CPU statevector circuit with three float32 UCCSD angles
and complex128 simulation. From the fixed initialization it reports an
initial energy near -1.1347755088 Ha and reaches within 1e-6 Ha of
-1.1372701747 Ha after 160 Adam steps. The
[full-register example](FERMION_H2_UCCSD_ENCODINGS.md) uses the same Adam loop.

Choose the -1 eigenvalue of each JW spin parity, `Z0 Z2` and `Z1 Z3`. Each
spin has two modes, so these eigenvalues fix one alpha and one beta electron.
The reference occupation is `1100`; the existing reduction map embeds the
two free virtual-spin bits back into the full register. Neither the sector
nor its reference is inferred.

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
At two initial points the tests compare the lifted state with the original
four-qubit JW circuit, and check the lift against an independent occupation
index table. They compare energies, all three gradients and the parameters
and energy after one shared Adam update. Lifting is test-only; training
evaluates native two-qubit expectations.

This example covers only the fixed H2 fixture in JW. It does not convert
arbitrary circuits, select sectors automatically, or taper parity or BK.
It adds no stable API, dependency, or performance claim, and requires neither
OpenFermion nor Cirq.
