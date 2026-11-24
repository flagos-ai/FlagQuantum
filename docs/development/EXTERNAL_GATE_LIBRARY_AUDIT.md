# Serialized external gate ownership audit

## Scope and ownership

This batch binds the frozen `CirqGateAsBloq` constructor in
`cirq_interop/_cirq_to_bloq.py` and its direct serialized gate, exact decomposition,
explicit matrix, scalar phase and controlled dependency paths. The reference is
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`. Reference code is
unchanged. Classification is `single_device_fast_path`.

`ecosystem/cirq/_component_library.py` decodes an existing ExternalGatePayload
through Cirq's existing JSON resolver. External SDK values remain local to this
boundary and a per-compilation constructor context. Returned Programs contain
owned native registers, parameters, matrices and clean-work lifetimes. Compiler
does not import external constructors or run numerical kernels.

The adapter reuses `import_cirq_register_gate` for declared decompositions and
`import_cirq_register_optree` for already-admitted primitive or explicit matrix
leaves. It does not contract an algorithm root into a dense matrix. Exact scalar
phase reuses the existing zero-port carrier recipe. Source control and target
ports remain THRU, arrays are row-major and words are MSB first.

## Direct controlled dependencies

Full resolved functions key the external gate recipes. Generic Controlled can
reconstruct only an explicitly owned child. Other children may delegate to a
supplied independent owner. Collision-free control names and bit patterns reuse
the shared signature and control-specification owners.

Negative and grouped control constructors reuse the existing ControlledViaAnd
orchestration. That owner retains source predicate computation, positive-child
call and exact measured erasure. The external gate owner supplies the positive
child through the existing SDK importer and stored control adder. Default
fixed X/Y/Z/H providers remain available. No inverse of a measured instruction
stream is invented.

Binary gate wrappers with neither a decomposition nor a unitary leaf retain a
typed bodyless declaration. They do not become executable channels implicitly.
Physical binding explicitly refuses their missing component body. Source
ResetChannel tensor refusal remains distinct from its logical declaration.

## Verification and boundaries

Standalone gate payloads compare source signatures, serialized depths zero and
one, full complex operators, exact global phase, owned work and IR round trips.
Symbolic binding is executed with source-package imports blocked. Generic,
negative and grouped external controls compare full operators and restored work.
Stored programmable arrays exercise arbitrary one- and two-qubit local rotation
and interleaver payloads through the ordinary append infrastructure.

The original Cirq JSON serializer cannot encode a `sin` expression in the gate
payload. Its TypeError is retained. A serializable symbolic parameter is used
for constructor round trips. Existing direct native parameter expressions retain
their separately accepted nonlinear execution path.

The original two-qubit arbitrary-gate array tensor path fails on non-THRU
`C[And]`. The native stored execution is compared against the original available
Cirq execution capture for that domain. This does not claim parity with the
failing generic tensor decomposition. Local matrix leaves and their synthesis
costs remain separate. Unsupported custom JSON resolvers, nonbinary physical
ports and unowned generic child controls retain explicit failures.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 129 cases without skips. Related local
integration passes 145 cases. All ten unchanged original `_cirq_to_bloq_test.py`
tests pass in the existing Cirq 1.6 compatibility environment. The native payload
and stored-array regression uses the current Cirq 1.7 environment. Four frozen
interop blobs, Qualtran 0.7.0 and actual Kickmix 0.1 Simulator are verified.
Strict mypy checks both new Python files. Black, Ruff, Ecosystem scope, language,
architecture, multi-level IR and API checks pass. Dependency analysis exactly
matches the 720 existing findings.

| Payload | Actual native SDK-import primitives | Independent original QEC model |
| --- | --- | --- |
| CXPow, exponent -0.375 | Two Ry, four X, four phase, one cphase; two used wires | One And, three Cliffords, one rotation, one measurement |
| One-qubit matrix | Six phase, one Rx, four X; one used wire | One rotation |
| Two-qubit matrix | 28 phase, four Rx, ten X, three cphase, four SX; two used wires | Three And, three Cliffords, 13 rotations, three measurements |
| Scalar phase | Two X, two phase, one restored carrier | Zero QEC gates |

These measured streams use the existing SDK importer. They are not an equality
claim against the full Qualtran named-gate decomposition. In particular, native
cphase remains atomic in this path while the named CZPow owner has its explicit
And/phase/erasure decomposition. Linking named-gate Programs and their helper
closures into this payload path is the next required fidelity step. The generic
constructor coverage counter is therefore unchanged by this batch.
