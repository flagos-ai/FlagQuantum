# Observables

This package owns the public mathematical observable model and the output
requests accepted by `fq.run` and `fq.plan`. It lowers those values to Core IR
measurement nodes; it does not execute numerical kernels, choose resources, or
adapt providers.

Start with `fq.expectation(fq.X(0) @ fq.X(1))` or
`fq.probabilities(qubits=(0, 1))`. Run the focused tests with:

```bash
pytest tests/api_contract/test_observable_outputs.py
```

Add a public output kind only when the Runtime can implement the same semantics
across its claimed backends and can fail closed elsewhere.

## Second-quantised operator algebra

`fermion.py` and `boson.py` build operators out of ladder operators and keep
them canonical, so a caller never applies an anticommutation or commutation
relation by hand.

`fermion.py` owns `FermionOperator` over spin orbitals. Modes are occupied below
the Fermi level, the canonical order is the one the anticommutation relations
impose, and `create`, `annihilate`, and `number` build the operators a chemistry
workload needs. Operators are normal-ordered against the vacuum, so equality is
structural and `is_hermitian` is exact rather than a tolerance test.
`jordan_wigner` is the one fermion-to-qubit transform: it maps a Hermitian
operator to the `Observable` the existing statevector path already measures,
which is how a fermionic observable acquires an expectation value.

`boson.py` owns `BosonOperator` over modes whose occupation is unbounded. The
same canonical order is the Poincare-Birkhoff-Witt basis, position and momentum
are built from it, and `to_matrix` evaluates an operator as a dense matrix over
a Fock space whose truncation dimension the caller names -- the algebra has no
finite matrix representation of its own, so no default dimension is assumed. The
dense form is a reference and a test oracle rather than a production path and is
refused above `DEFAULT_DENSE_MATRIX_BYTES`.

Neither module executes anything: there is no bosonic or photonic backend, no
continuous-variable simulator, and no circuit-level exponential. Run the focused
tests with:

```bash
pytest tests/unit/test_observables_fermion.py tests/unit/test_observables_boson.py
```
