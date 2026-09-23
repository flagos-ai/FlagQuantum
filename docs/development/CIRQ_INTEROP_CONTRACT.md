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

## Measurement conversion contract

The measurement extension implements a deliberately narrow static mapping. The
capability matrix records Cirq `measurements` as `partial`: terminal measurement
gates convert in both directions, while runtime measurement execution and
sampling remain `out_of_scope` because they require an execution plan rather
than a circuit conversion. IR `MeasurementNode` execution requests continue to
fail closed with `measurement_not_represented`.

The first version admits only terminal measurement:

- a `cirq.MeasurementGate` is expanded into one FlagQuantum `measure`
  instruction per measured qubit, so every measured qubit owns exactly one
  classical bit;
- the measured qubit order is the gate's own `qubits` source order, and the
  expanded instructions for one key keep that order;
- the Cirq measurement key is preserved verbatim as a non-empty string without
  rewriting, and the key must be unique in the circuit: two measurement gates
  that share a key are rejected because the resulting instruction metadata would
  be ambiguous;
- every admitted measurement must follow all non-measurement operations of the
  circuit, so mid-circuit measurement is rejected;
- the FlagQuantum instruction carries the key and the assigned classical bit in
  FlagQuantum IR instruction metadata, which is the metadata owner; and
- the adapter owns the classical-bit assignment: indices start at zero, the
  measured qubits of one key receive a contiguous block in source order, the
  counter is circuit-global and monotonic, and no index is ever reused.

The same first version rejects every measurement form that FlagQuantum IR cannot
represent losslessly: `invert_mask`, `confusion_map`, empty keys, duplicate keys,
measured qids whose dimension is not two, and any `MeasurementGate` form the
adapter does not recognize. Rejection is fail-closed with the dedicated issue
codes recorded in `[measurement_contract].rejection_issue_codes`.

Cirq gate and qubit objects stop at `flagquantum.ecosystem.cirq`, so the key,
qubit order, and classical-bit assignment are converted to FlagQuantum strings,
integers, and tuples before an instruction enters IR. The extension introduces
no second registry and no public API: it extends the existing
`[measurement_contract]` table of `contracts/cirq-interop-contract.toml` and is
enforced by the existing Cirq contract checker.

The integration suite demonstrates these acceptance criteria:

1. terminal measurement round-trips with its key, qubit order, and classical
   bits;
2. a multi-qubit gate expands to one instruction per qubit in source order;
3. each rejected form fails closed with its recorded issue code;
4. the classical-bit counter never reuses an index within a circuit; and
5. no Cirq or SymPy object is retained in core IR.
