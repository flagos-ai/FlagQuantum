# Observables

This package owns the public mathematical observable model and the output
requests shared by circuit and CPU dynamics workflows. Pauli requests lower to
existing Core IR measurement nodes for `fq.run` and `fq.plan`; full-space matrix
expectations use `flagquantum.lindblad`. This package does not execute numerical
kernels, choose resources, or adapt providers.

Start with `fq.expectation(fq.X(0) @ fq.X(1))` or
`fq.probabilities(qubits=(0, 1))`. Run the focused tests with:

```bash
pytest tests/api_contract/test_observable_outputs.py
```

Add a public output kind only when the Runtime can implement the same semantics
across its claimed backends and can fail closed elsewhere.

## Full-space matrix expectations

`fq.expectation(matrix, name="population")` accepts an ordinary dense numeric
PyTorch tensor. The matrix is square, nonempty and covers the entire declared
state space; nonempty qubit selectors are refused. The existing CPU Lindblad
planner checks dimension, finiteness and Hermiticity and captures values without
detaching the live gradient graph. Product ordering or explicit basis labels
remain the caller's existing state-space declaration. Rebuild the plan after
optimizer updates. JSON replay retains numeric content, not the graph.

Only expectation requests accept tensors. Matrix requests compare by tensor
object identity plus metadata; equal clones remain distinct requests, while
equal numeric plans keep the existing value-based identity. Pauli request
comparison and hashes are unchanged. Circuit and remote paths raise
`CapabilityError` directing the caller to Lindblad before execution/submission;
there is no implicit Pauli conversion, embedding or new CircuitIR instruction.

Run `python -m examples.matrix_expectation_dynamics` for the three-level model.
Expected CPU output:

```text
Leakage expectation: 0.11757891
Coupling derivative: 0.32210884
Circuit matrix output: refused; use flagquantum.lindblad
```

## Second-quantised operator algebra

`fermion.py` and `boson.py` build operators out of ladder operators and keep
them canonical, so a caller never applies an anticommutation or commutation
relation by hand.

`fermion.py` owns `FermionOperator` over spin orbitals. Modes are occupied below
the Fermi level, the canonical order is the one the anticommutation relations
impose, and `create`, `annihilate`, and `number` build the operators a chemistry
workload needs. Operators are normal-ordered against the vacuum, so equality is
structural and `is_hermitian` is exact rather than a tolerance.

Two fermion-to-qubit transforms map a Hermitian operator to the `Observable` the
existing statevector path already measures, which is how a fermionic observable
acquires an expectation value. `jordan_wigner` writes mode `k` as
`Z[0] ... Z[k-1] sigma[k]`, so the string runs over the modes below the factor;
`parity_encoding` writes the same operator with an `X` string over the modes above
it, which is the Jordan-Wigner image conjugated by the change of basis from
occupation to parity. The two are the same observable in two bases and differ in
where the image is local, not in how wide its widest term is: the reduction in
Pauli weight that motivates the Bravyi-Kitaev and ternary-tree encodings follows
from neither, and both of those remain absent.

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
