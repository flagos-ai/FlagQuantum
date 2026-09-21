# Qiskit arithmetic parameter-expression assimilation

## Capability slice

FlagQuantum imports Qiskit gate parameters that simplify to the owned v1
arithmetic subset: named parameters, real numeric constants, addition,
multiplication, and negation. This closes the previous asymmetric boundary in
which FlagQuantum could export the same expressions to Qiskit but could not
import them back.

Functions, powers, division by an unbound parameter, complex constants,
duplicate parameter names, and symbols outside the Qiskit parameter set fail
closed with `unsupported_parameter_expression`. Control flow, register-layout
flattening, and custom multi-qubit unitary behavior are unchanged.

## Reference and provenance

- Reference SDK: Qiskit 2.0.x and 2.5.x, the repository's existing certified
  lanes.
- Public boundary: `qiskit.circuit.ParameterExpression.sympify()`.
- Qiskit 2.0 release notes explicitly identify SymPy as the optional dependency
  required by `sympify()`:
  <https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.0>.
- Qiskit API documentation defines `sympify()` as returning a SymPy-equivalent
  expression:
  <https://quantum.cloud.ibm.com/docs/api/qiskit/2.3/qiskit.circuit.ParameterExpression>.
- Qiskit and FlagQuantum are Apache-2.0 licensed. SymPy is BSD-licensed. No
  upstream source code is copied; the adapter consumes public objects only.

SymPy is part of the optional `qiskit` extra because Qiskit 2.x no longer
installs it by default. It remains isolated to the ecosystem adapter and can be
removed if Qiskit exposes a stable, dependency-free expression tree in a future
certified release.

## Semantic contract

- Parameter identity follows the existing unique-name contract.
- Qiskit performs symbolic simplification before conversion. FlagQuantum owns
  the resulting expression tree and retains no Qiskit or SymPy objects.
- Real numeric constants are converted to Python `float` values.
- SymPy `Add` and `Mul` nodes are folded deterministically into FlagQuantum
  `ParameterExpression` nodes. Negation and subtraction therefore retain their
  mathematical meaning even if their tree shape differs from the source.
- Unsupported nodes are rejected before an instruction enters `CircuitIR`.
- Binding the imported FlagQuantum IR and binding the round-tripped Qiskit
  circuit must produce the same gate parameter within `pytest.approx`'s default
  scalar tolerance.

## Verification and maturity

The focused integration tests cover a two-parameter expression with scaling,
negation, addition, FlagQuantum binding, and Qiskit round-trip binding. A sine
expression is the explicit negative case. Existing gate, wire-order,
measurement, statevector, and fail-closed tests remain the wider regression
boundary.

This slice remains **experimental**, matching the Qiskit adapter. Local evidence
uses Qiskit 2.5.2; CI owns the existing 2.0.x and 2.5.x certification lanes.
