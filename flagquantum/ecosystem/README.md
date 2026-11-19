# Ecosystem interoperability

`flagquantum.ecosystem` is the interoperability boundary. It translates objects
from optional external frameworks into FlagQuantum-owned `CircuitIR` and
translates owned programs back only when explicitly requested. Conversion
reports make every unsupported or lossy semantic difference visible.

Interop does not define a second canonical IR, compile programs, implement
numerical kernels, or let framework objects enter Core, Compiler, Runtime, or
Simulation. An explicitly selected local external simulator may execute an
owned program inside its framework package and must return a FlagQuantum-owned
result. Importing this package and listing its adapters must not import an
optional framework.

## Where to start

- `contracts.py`: framework-neutral adapter and conversion-report protocol.
- `registry.py`: immutable, lazy adapter discovery.
- `conformance.py`: common round-trip and rejection checks.
- `_semantic_parity.py`: deterministic circuits shared by optional SDK parity lanes.
- `<framework>/conversion.py`: external object to/from `CircuitIR` conversion.
- `<framework>/models.py`: adapter-local results and errors.
- `<framework>/adapter.py`: the small implementation of the common protocol.
- `<framework>.run`: the consistent explicit entry point when that framework
  has a qualified local execution bridge.

Framework-specific local execution entry points use
`flagquantum.ecosystem.<framework>.run` and reuse native `fq.run` vocabulary:
`outputs`, `ExecutionOptions`, `samples`, and `counts`. Remote services and
provider hardware remain owned by Remote; an Ecosystem runner must not become
an implicit router or fallback path.

## Ten-minute change path

For a small adapter conversion change, modify one framework adapter, add its
round-trip or explicit-rejection case, and run:

```bash
python -m pytest tests/team/ecosystem tests/api_contract/test_open_source_golden_paths.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Framework-specific dependencies remain optional and isolated under their
adapter package. Changes to the candidate-stable interop protocol, signatures,
issue schema, or registry semantics require the public API change process;
ordinary framework mappings should not extend that shared protocol.


## Native construction source (private)

`_qmod/` adapts a bounded, pinned native Qmod subset into verified Compiler
construction records before gate-level IR exists. This private handoff differs
from the public CoreIR adapter protocol; it does not extend that protocol or
register another framework. Compiler retains lowering/selection and Runtime
retains execution. See its [contributor guide](_qmod/README.md) and
[separate compatibility counts](../../docs/roadmap/CS07_QMOD_PARSER_DELIVERY.md).
No public native-source execution entry is admitted by this private work.

## Logical-resource estimator export

`resource_estimator.py` converts existing declared logical resources to the
four-field Microsoft known-count schema. It imports no optional SDK and performs
no physical estimation. The declared width controls whether the exported count
includes work qubits.

```python
from flagquantum.compiler._component_trace.resource_model import GateCost, ModeledResources
from flagquantum.ecosystem.resource_estimator import microsoft_logical_counts

counts = microsoft_logical_counts(ModeledResources(GateCost(t=100), 17))
assert counts == {"numQubits": 17, "tCount": 100, "rotationCount": 0, "rotationDepth": 0}
```

The complete optional local-estimator workflow is in
`examples/notebooks/physical_resource_interop.ipynb`. SDK result and widget objects
remain inside that interoperability example. Test a conversion change with
`python -m pytest tests/unit/test_resource_estimator_counts.py -q`.

## Bound register Program conversion

`cirq._register_program_export.export_bound_register_cirq` translates a
Compiler-bound finite Program with consecutive input wires into a Cirq circuit
and shaped output Qids. It replays external manager lifetimes and transfers
new output leases to the caller. Allocation policy stays in Compiler. Follow
[the executable example and ownership rules](../../docs/development/CIRQ_REGISTER_PROGRAM_EXPORT.md)
when modifying this internal adapter. Run
`tests/unit/test_cirq_register_program_export.py` and
`tests/test_cirq_register_export.py` to cover direct and gate-frame conversion.

The register importer recursively resolves concrete SDK decompositions while
reserving sibling operands through the shared interop qubit manager. Native
typed definitions take precedence. Decompositions require clean work. Concrete matrix leaves use the existing owned matrix payload and remain atomic
unless a physical decomposition is supplied. Run `tests/unit/test_cirq_register_leaf_closure.py`
with the frozen reference environment to check wrapper, lifetime and append
round trips.

Private `RegisterCirqGate` accepts an optional first-operand `gate_label` and
preserves it under power and adjoint. Named register import accepts scalar Qids
for scalar one-bit ports while retaining shape and alias checks. The complete
native tutorial is `examples/notebooks/register_gates.ipynb`; its reusable
operation is `examples/algorithms/register_gates.py`. Run
`tests/unit/test_register_gate_tutorial.py` for source stream, complex operator,
lease, diagram and append comparisons. See
[the source audit](../../docs/development/REGISTER_GATE_SOURCE_AUDIT.md) for
zero-width and wider typed-control boundaries.

### Real parameter expressions and scalar phases

Cirq and Qiskit conversion translate the approved sin/cos/exp/log/pow/div
expressions directly into Core ParameterExpression values. SDK symbolic objects
stop at this boundary. Circuit binding validates real domains and preserves
Torch scalar gradients. Cirq symbol discovery preserves assumptions while
rejecting distinct symbols with the same native name across supported gates.

Cirq exp(i * real_angle) and Qiskit symbolic global phase use Compiler's existing
scalar phase emitter. Static Cirq requires a declared wire; register conversion
can allocate its clean scalar carrier. One- and two-control binary patterns
preserve exact phase and operand order. Generic X/Y/Z/ZZ/CZ eigen-gate shifts preserve both symbolic fields and the
exact scalar phase through the existing primitives. Run tests/unit/test_nonlinear_sdk_parameters.py
and tests/unit/test_symbolic_global_phase.py for the shortest executable examples.

## Fixed-point external values

`fixed_point.py` owns optional real-object conversion to the existing native
integer and bit representation. Its lazy dependency and source-conformance
boundary are documented in
[the conversion guide](../../docs/development/FIXED_POINT_REAL_CONVERSION.md).
