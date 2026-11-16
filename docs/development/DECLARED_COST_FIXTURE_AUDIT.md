# Declared cost fixtures

The Qualtran v0.7.0 fixtures in Q-0491 and Q-0492 were verified at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The unchanged source test
passes. The native fixture is confined to tests and uses the existing
symbolic call graph, qubit and QEC cost queries.

The reference example matches all six debug edges in order. T and its
adjoint have separate leaf identities and contribute 20 calls apiece.
The QEC model totals 40 T, 20 Clifford and 100 Toffoli, with a 100-qubit
root declaration. This fixture is entirely analytical and has no
physical circuit, measurements or workspace.

The duplicate-key check verifies that immutable input sequences retain
their identity, while their mappings take the last value for repeated
keys. A symbolic T count survives graph expansion, and independent
static overrides return 11 T and eight qubits. A zero-width signature
is empty. The source uses an open set of cost keys. This test fixture
adapts the two checked metrics to the existing native typed queries.

The focused graph and resource regression passed 236 tests. Strict
typing, formatting, team scope, architecture, IR and public API
checks passed.
