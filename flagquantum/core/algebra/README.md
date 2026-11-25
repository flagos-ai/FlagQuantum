# Complex operator values

Owns canonical complex Pauli arithmetic. Uses only the standard library. Does
not import Observable, algorithms, circuits, runtime, or external frameworks.
Existing FermionOperator remains authoritative in observables.fermion.

Entry points are PauliOperator, PauliTerm and pauli in pauli.py. For example,
`pauli(factors=((0, "x"),)) * pauli(factors=((0, "y"),))` is `1j * Z(0)`.
Run `python -m pytest tests/unit/fermion/test_encoding.py -q` after a change.
