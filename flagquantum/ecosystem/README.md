# Ecosystem interoperability

The packed-oracle component-library adapter converts Hubbard coefficients to
owned angles with the shared Simulation coefficient owner, then delegates call construction
and scoped exact inverses to Compiler. It records explicit selection/junk/system
roles and reuses the existing scalar carrier for uncontrolled SELECT. It neither
executes numerical states nor infers missing child semantics from resource counts.
See the [constructor audit](../../docs/development/HUBBARD_PREPARATION_CONSTRUCTOR_AUDIT.md)
and `tests/unit/test_hubbard_prepare_source_constructors.py` for the smallest
source round-trip, source-free append and complete complex-branch scenarios.

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

The private `_sparse_walk_component_library.capture_sparse_walk_context` captures
locked sparse chemistry source children into native exact pairs and an explicit
retained bank environment. SDK objects end at this boundary. The native context
has extra signal ports and does not impersonate the narrower source constructor
ABI. Run `tests/unit/test_sparse_walk_component_library.py` for the original
65-row metadata/binding witness, small complex joint inverses and SDK-free
ordinary `circuit.append(op)` construction.

## Where to start

The private `_fourier_measurement_library` owns the original MeasureQFT
declaration shared by RSA and ECC constructor adapters. It retains the source
unused-CBit failure; explicit numerical readout remains an Algorithms workflow.
The `_elliptic_curve_phase_library` translates point/window fields and delegates
ordered composition to Compiler. Use
`tests/unit/test_elliptic_curve_phase_components.py` for source-library round trips,
point-power operands, dirty exits and SDK-free selected-child execution.

- `contracts.py`: framework-neutral adapter and conversion-report protocol.
- `registry.py`: immutable, lazy adapter discovery.
- `conformance.py`: common round-trip and rejection checks.
- `_component_library.py`: explicit bitwise library constructor binding to the
  existing numerical constant codec. Compiler receives owned MSB patterns.
  Run `tests/unit/test_legacy_word_factories.py` for the serialized-library
  through ordinary append scenario and encoder replacement.
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
expressions through the shared Compiler symbolic-scalar converter into Core
ParameterExpression values. Approved acos/floor expressions use the same path.
Cirq supports both; Qiskit maps acos through its public arccos operation and
explicitly refuses unbound floor, which can be numerically bound before export.
SDK symbolic objects stop at this boundary. Circuit binding validates real
domains and preserves native Torch scalar gradients. Symbol discovery rejects
distinct symbolic identities with the same native name across supported gates.

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

The internal Cirq multiplication bridge converts concrete external declarations
into the existing owned register Programs. Compiler retains arithmetic and
measured cleanup. The bridge handles SDK allocation, output aliases and
measurement keys. Use `tests/unit/test_multiplication_foreign_file.py` for the
eight-class capture, physical rebinding, serialization and append workflow.

## Live typed component graphs

`cirq._register_graph.import_register_graph` converts a live typed source graph
at the optional SDK boundary. Actual source occurrences and connections feed
the existing owned graph builder; only nominal signatures use the register
codec. Declared Allocate/Free and Split/Join map to the existing
Compiler definitions, preserving clean versus borrowed origin and typed word
ports. Other leaves use the existing Cirq conversion. No source constructor is
loaded from a serialized name. Distinct live leaves retain their own physical
callbacks even when their constructor declarations serialize identically.
Montgomery moduli omitted by the source codec are retained from live metadata.
Original constructor-serialization failures remain observable independently;
they do not prevent importing a valid live graph.

Use this graph path when the source declares borrowed lifetimes. The flat
`CircuitRegisterEmitter` still describes clean workspace. Native Program
binding accepts a separate borrowed pool, and normal `circuit.append(op)`
consumes its lowered instructions. Run `tests/unit/test_cirq_register_graph.py`
for original single-/three-axis examples, nested prefix folding, clean private
children, dirty echo children, every small measurement branch and SDK-free
execution after conversion. Compiler owns graph scheduling, allocation and
binding, while the source resource models remain independent.

The existing Cirq notebook helper retains caller-relative file resolution and
its execution defaults. It delegates optional Jupyter execution to
`flagquantum.testing.notebook_execution`, imported only when executing a
notebook. The adapter owns external file selection and conversion, while the
shared Testing helper owns kernel lifecycle. Run
`tests/unit/test_cirq_notebook_resources_file.py` after changing that delegation.


## Native command argument helpers

`_command_arguments.py` owns first-match argv scanning, typed flags, enum
selection and scoped file acquisition. `_command_numbers.py` owns the frozen
signed-integer grammar and direct host-C float32/64 parsing. Existing KMX
commands reuse the integer callback while argparse retains orchestration and
help. These private helpers carry no Core, Compiler or Simulation state.

Run `python -m pytest tests/unit/test_command_arguments.py -q` for the shortest
change path. `tests/unit/fixtures/generate_command_arguments.py` regenerates
frozen-source observations. See
`docs/development/COMMAND_ARGUMENTS_AUDIT.md` for exact ownership, original
integer wrap aliases and the explicitly open host-specific C file-mode boundary.

## Unified physical-format commands

`kmx_command.py` selects the existing format, sampling or drawing owner with
lazy imports. The complete dispatcher accepts all eight original commands.
For a short generation and conversion round trip, run

```bash
python -m flagquantum.ecosystem.kmx_command circuit --kind iadd --n 3 --out /tmp/add.kmx --validate
python -m flagquantum.ecosystem.kmx_command kmx2kmb --in /tmp/add.kmx --out /tmp/add.kmb
python -m flagquantum.ecosystem.kmx_command count_operations --in /tmp/add.kmb
```

Run `tests/unit/test_kmx_command_dispatch.py` after changing selection or
argument forwarding. Its workflow invokes every owning command and executes the
converted circuit through `circuit.append(op)`. Source-boundary details are in
`docs/development/MAIN_COMMAND_AUDIT.md`.

## Exact phase object conversion

`_phase_angle.phase_angle_from_half_turns` converts an external Python value
through Fraction normalization and ties-to-even rounding into the existing
Compiler PhaseAngle128. Decimal strings and binary floats retain their distinct
exact values. The Compiler double/decimal constructors retain their existing
policies. The helper is the common boundary used when creating raw phase records
or seeding a packed phase accumulator.

```python
from fractions import Fraction
from flagquantum.ecosystem._phase_angle import phase_angle_from_half_turns

angle = phase_angle_from_half_turns(Fraction(1, 4))
assert angle.half_turns() == Fraction(1, 4)
assert angle.clifford_hierarchy_level() == 3
```

Run `tests/unit/test_phase_angle_objects.py` for both frozen source consumers,
full raw words, binary round trips, native phase composition and the normal
circuit.append(op) complex-state workflow. See
`docs/development/PHASE_ANGLE_OBJECT_AUDIT.md` for the complete source scope.

## Private physical observation conversion

`_physical_basis_session.PhysicalBasisSession` resolves native physical operand
identities, register aliases, external arrays, exact phase inputs and debug
callbacks. Simulation `_basis_trajectories` owns all numerical state and random
streams. The adapter reuses existing Compiler metadata, operand binding and
physical statistics. It imports no reference extension. See
[the complete session mapping](../../docs/development/PHYSICAL_BASIS_SESSION_AUDIT.md).

`_physical_circuit_files` converts Python paths and caller-owned streams using
existing Compiler KMX/KMB serializers. It returns native immutable circuit values
and leaves supplied streams open. It owns no parser, execution kernel or quantum
workspace. Run `tests/unit/test_physical_circuit_files.py` for source byte
round trips, failure ordering, resource projections and normal-append/all-branch
consumer replacement. The complete main-registration disposition is recorded in
`docs/development/PHYSICAL_MAIN_BINDING_AUDIT.md`.

`_component_library.arithmetic_library_factories` binds arithmetic constants
through the same Simulation stored-value codec as bitwise constructors. Compiler
owns construction and Simulation owns numerical encoding.

The Cirq external-gate library factory owns JSON payload conversion and scoped
explicit adjoints in `cirq/_component_library.py`. Exact inversion stays in
Compiler's component-call owner. Generic external controlled adjoints retain
their immediate child boundaries in `cirq/_controlled_gate_inverse.py`; its
qualified SDK execution views do not replace named measured owners. Atomic
power children reuse the measured positive CZPow owner; nested explicit
adjoints rebuild directed child calls rather than borrowing a forward SDK recipe. Feedback replacement replay remains local to
already converted bodies. Start with
`tests/unit/test_cirq_gate_library_factories.py` and the
[external-gate audit](../../docs/development/EXTERNAL_GATE_LIBRARY_AUDIT.md).
For cost-only inspection, `cirq.resource_keys.cirq_mapped_gate_resource_key`
returns native model data with the original opaque-atom precedence; pass that
key to the existing `query_qec_gate_cost`. A missing model stays unavailable.
The existing recursive decomposition resource entry has a different grain and
is unchanged. Controlled SDK scalar coefficients reuse the existing native global-phase cost
owner, preserving the principal angle and predicate controls. Uncontrolled
phase stays free. Neither model query emits gates or makes a body executable.
