# Four-mode H2 UCCSD across three encodings

The existing FCIDUMP training example accepts `jordan_wigner` (the default),
`parity` or `bravyi_kitaev`:

```bash
python -m examples.algorithms.fermion_fcidump_training --encoding parity
```

It uses the same H2 integrals, UCCSD angle order and Adam loop for all three
choices. The fixture has two spatial orbitals and two electrons. The example
reports a reference determinant energy of about -1.1166843871 Ha and reaches
within 1e-6 Ha of the bounded two-particle reference at -1.1372701747 Ha.
The final particle count is two. This is evidence for the fixture and its
fixed initialization.

The existing UCCSD circuit prepares the state in occupation order. The example
uses the encoded unit occupations as the columns of a binary map. For parity
and Fenwick Bravyi–Kitaev at this width, each encoded bit is its own occupation
bit XOR selected lower-index occupations. Emitting the CNOTs in descending
target order preserves those source bits until they are no longer needed.
The Hamiltonian and particle-number observable use the matching explicit
encoding. For the test-size dense reference, the code selects the two-particle
sector in physical occupation order and maps those indices before slicing
the encoded matrix. Counting set bits directly in an encoded index would
select the wrong sector.

The focused integration test checks every amplitude against an independent
occupation-bit permutation, compares energy and each float32 angle gradient
with JW at two parameter points, and compares updated parameters and energy
after one Adam step. CLI tests check the shared training loop and its
finite-difference report in each encoding. The FCIDUMP isolation lane also
runs the BK training path with OpenFermion, Cirq and external chemistry
imports blocked.

This is a full four-qubit, exact CPU statevector example. It preserves the
existing float32 angle convention and uses complex128 simulation. The dense
reference is bounded inspection with exponential storage in the circuit
width. The example does not define a general circuit conversion or tapered
ansatz, and adds no Stable API or OpenFermion/Cirq dependency.
