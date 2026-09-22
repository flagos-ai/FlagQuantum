# Cirq interoperability contract

FlagQuantum will support `cirq.Circuit` through an optional adapter owned by
`flagquantum.ecosystem.cirq`. Cirq objects stop at that package boundary and are
converted immediately to FlagQuantum IR. Core, compiler, runtime, and simulation
modules must not import Cirq or expose Cirq types.

The first integration change established the boundary before implementation.
The implemented adapter remains experimental and does not promise runtime
execution through Cirq. The machine-readable contract fixes the initial gate
subset, contiguous `LineQubit` mapping, explicit statevector order, the
supported symbolic arithmetic subset, and fail-closed handling of unsupported
Cirq features.

## Dependency decision

The adapter needs `cirq-core`, rather than the `cirq` meta-package, because only
the circuit model, gate types, protocols, and reference simulator are required.
It is isolated in the `cirq` extra and certified against 1.6.1 and 1.7.0 on
Python 3.12. Cirq is Apache-2.0 licensed; transitive packages remain visible to
the repository's audit and SBOM jobs. The dependency can be removed by deleting
the optional adapter and its extra without changing FlagQuantum IR or stable
core APIs.

The replacement interface is the FlagQuantum interoperability adapter protocol.
If Cirq changes its gate model or maintenance status, the adapter can pin a safe
lane, translate through public Cirq protocols, or be retired while other
adapters continue to consume the same FlagQuantum-owned IR.

## Implementation acceptance

The implementation provides:

1. convert supported `cirq.Circuit` programs in both directions;
2. reject unsupported features unless loss is explicitly authorized and
   reported;
3. prove statevector equivalence with an explicit qubit order;
4. test both certified Cirq versions in the optional CI lane;
5. prove importing `flagquantum` does not import Cirq; and
6. register the adapter publicly only after integration and conformance tests
   pass.

## Symbolic parameter extension contract

The adapter preserves supported symbolic rotation parameters across the Cirq
boundary without changing the public API.

The adapter will discover symbols through `cirq.parameter_symbols` and use
`cirq.resolve_parameters` as the Cirq-side binding reference. Cirq and SymPy
objects remain confined to `flagquantum.ecosystem.cirq`; the FlagQuantum-owned
representation is `Parameter` or `ParameterExpression`, identified by a unique
parameter name.

The first implementation supports a deliberately small arithmetic subset:

- symbols and finite real constants;
- addition and multiplication; and
- unary negation, with subtraction represented as addition plus negation.

Functions, powers, complex constants, division by an unbound parameter, and
symbols outside the expression's reported parameter set fail closed with the
`unsupported_parameter_expression` issue code. Import must reject the
expression before an instruction enters IR, and export must reject it before a
Cirq operation is created. Supported SymPy additions and multiplications are
canonicalized into deterministic left folds before conversion.

The integration suite demonstrates round-trip preservation for expressions
with more than one symbol and binding equivalence at multiple assignments:
resolving a Cirq circuit with `cirq.resolve_parameters` must produce the same
bound gate values as binding the corresponding FlagQuantum circuit. It also
proves deterministic failure for every unsupported expression family and that
no Cirq or SymPy object is retained in core IR.

The boundary relies only on Cirq's public parameter protocols:

- <https://quantumai.google/reference/python/cirq/parameter_symbols>
- <https://quantumai.google/reference/python/cirq/resolve_parameters>
- <https://quantumai.google/reference/python/cirq/ParamResolver>
