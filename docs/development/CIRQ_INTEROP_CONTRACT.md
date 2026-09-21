# Cirq interoperability contract

FlagQuantum will support `cirq.Circuit` through an optional adapter owned by
`flagquantum.ecosystem.cirq`. Cirq objects stop at that package boundary and are
converted immediately to FlagQuantum IR. Core, compiler, runtime, and simulation
modules must not import Cirq or expose Cirq types.

The first integration change establishes the boundary before implementation. It
does not add a public adapter or promise runtime execution through Cirq. The
machine-readable contract fixes the initial gate subset, contiguous
`LineQubit` mapping, explicit statevector order, bound-real parameter policy,
and fail-closed handling of unsupported Cirq features. A later ecosystem change
must implement the conversion and all planned verification paths before changing
`implementation_status` or `public_api_available`.

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

The implementation PR must:

1. convert supported `cirq.Circuit` programs in both directions;
2. reject unsupported features unless loss is explicitly authorized and
   reported;
3. prove statevector equivalence with an explicit qubit order;
4. test both certified Cirq versions in the optional CI lane;
5. prove importing `flagquantum` does not import Cirq; and
6. register the adapter publicly only after integration and conformance tests
   pass.
