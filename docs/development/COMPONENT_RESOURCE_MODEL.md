# Component resource estimates

The component resource model can estimate an operation before generating its
IR. Each node declares exclusive logical gate counts and its local peak
width. Each child edge declares a call multiplicity and the bystander qubits
live during that call. Counts add across calls. Width takes the maximum
scheduled width, so sequential reuse does not multiply the qubit count.
A leaf can declare a cost without an executable decomposition.

The model retains T, Toffoli, controlled swap, And, Clifford, rotation and
measurement counts separately. The default T-equivalent conversion uses
four T per Toffoli, controlled swap or And, and eleven per generic rotation.
These conversion factors are configurable. The result describes logical
resources and has no hardware timing, physical width or fidelity estimate.

The dense and sparse rotation preparations supply concrete model builders.
This example needs no Qualtran installation or circuit generation.

```python
from flagquantum.algorithms.primitives.rotation_resource_model import (
    sparse_rotation_preparation_cost,
)
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

preparation = SparseRotationPreparation((2, 1), (1 + 1j, 1 - 1j), 3, 8)
estimate = estimate_component_resources(
    sparse_rotation_preparation_cost(preparation, target_width=4)
)
assert estimate.gates.total_t_count() == 92
assert estimate.gates.measurement == 11
assert estimate.peak_logical_qubits == 22
```

For the same sparse preparation, the emitted native IR has 100 T/T†,
25 measurements and 28 declared qubits (12 data and 16 clean work qubits).
The model uses the original abstract gradient addition. The measured adder
in native IR has a different resource profile. The existing
`estimate_resources` entry point continues to describe static instructions
and basis conversion, with its existing data-dependence rules.

## Beverland surface-code estimate

A declared logical cost can feed a surface-code model before any circuit
is generated. The experimental QEC estimator applies the pinned Qualtran
v0.7.0 Beverland logarithmic rotation model, with one third of the total
error budget reserved for synthesis. It estimates T-state equivalents,
a minimum number of time steps, and the physical-qubit footprint of the
fast data block. Code distance defaults to the minimum-time scenario.
Pass `time_steps` to estimate distance at another time budget and
`rotation_layers` to replace the rotation-count proxy for layer depth.

```python
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.qec.resources.beverland import estimate_beverland_resources

logical = estimate_component_resources(select_thc_cost(2, 4, 8, 1))
report = estimate_beverland_resources(
    logical, error_budget=1e-3, physical_error=1e-4
)
assert logical.peak_logical_qubits == 24
assert report.t_states == 472
assert report.minimum_time_steps == 354
assert report.code_distance == 7
assert report.data_block_physical_qubits == 6174
```

The 6,174 physical qubits cover the 63 fast data-block tiles, including
their layout overhead. A full machine estimate also needs factory sizing
and throughput. An estimate with zero modeled time steps and no override
leaves distance and footprint undefined. Read T-state demand separately
from the number of T and T-dagger instructions emitted for a concrete
circuit. The fixed-precision THC model already declares its phase
addition costs, so this estimate applies no extra rotation synthesis to
those operations.

## Model and emitted instructions

For a generated circuit, call
`compare_component_resources(model, circuit_ir)` from
`flagquantum.compiler._component_trace.resource_comparison`. Its JSON report
separates the model from the emitted instructions. The IR section counts each
instruction once, including conditional instructions, without resolving a
measurement history. It reports the declared width, the number of qubits
touched by instructions, and output requests separately.

For example, continue with the preparation above:

```python
from flagquantum.compiler._component_trace.resource_comparison import (
    compare_component_resources,
)
from flagquantum.core.ir import CircuitIR

work, slots = preparation.requirements(4)
ir = CircuitIR(
    12 + work,
    preparation.emit(
        tuple(range(4)), tuple(range(4, 12)),
        tuple(range(12, 12 + work)), tuple(range(slots)),
    ),
)
report = compare_component_resources(
    sparse_rotation_preparation_cost(preparation, target_width=4), ir
)
assert report.emitted.t_count == 100
assert report.emitted.unconditional["measure"] == 25
assert report.emitted.conditional == {"cz": 25}
assert report.emitted.declared_qubits == 28
```

Reusable register models cover concrete unary QROM, Add,
AddIntoPhaseGrad, equality-driven basis permutations, QROAMClean and
SelectSwapQROM with clean or borrowed banks. The QROM model
handles bounded multidimensional axes and shaped target words. Its width
comes from the locked static formula, including the full selection pool
even when a uniform table uses less work. The phase-add model retains
the upstream abstract Toffoli count. The one-bit source Add reports
−3 Clifford gates and one-bit AddIntoPhaseGrad can report a negative
Toffoli count. These values reproduce source formulas for comparison.
The one-bit native Add emits one CX.

The QROAM measurement-erasure wrapper estimates its Toffolis from the
original table extent. The underlying leaf uses the padded extent. For
five rows, two-word forward banks, one control and one-word erasure blocks,
the wrapper estimates three Toffolis and the padded leaf estimates four.
Both report zero measurements in the locked cost declarations. The source
measurement-erasure leaf has no executable decomposition. These counts are
separate from the emitted reverse-lookup circuit.

Use `modeled_call_graph` from
`flagquantum.compiler._component_trace.call_graph` to inspect the declared
direct edge weights and weighted terminal calls. Structurally equal
components share a node. Ordered generalizers can merge or omit children.
A keep predicate or maximum depth can stop expansion. A shared node's first
depth-first visit determines its expansion, matching the locked source.
The terminal counts describe calls and omit exclusive gate costs declared
on interior nodes. Use the resource estimate for those totals.

Query a single metric with `query_component_cost` from
`flagquantum.compiler._component_trace.cost_query`. The built-in metrics are
`GATE_COST`, `PEAK_LOGICAL_QUBITS` and `SUCCESS_PROBABILITY`. A custom
`CostMetric` supplies a recursive rule and the identity returned for an
omitted component. The query first generalizes the node, then checks the
caller cache, checks explicit static overrides and finally evaluates the
recursive rule. A success probability declared on the component remains
the built-in probability rule's default override. Use each cache with a
single metric and a fixed generalizer and override configuration.

```python
from flagquantum.compiler._component_trace.cost_query import (
    PEAK_LOGICAL_QUBITS,
    query_component_cost,
)

model = sparse_rotation_preparation_cost(preparation, target_width=4)
assert query_component_cost(model, PEAK_LOGICAL_QUBITS) == 22
assert query_component_cost(
    model, PEAK_LOGICAL_QUBITS, static_costs={model: 24}
) == 24
```

Use `query_component_costs` for a table indexed by component and metric.
Each metric starts with its own empty cache. A row contains only the
metrics that visited it. For example, a declared success probability can
stop that query at the parent while gate and width queries still visit
its children.

```python
from flagquantum.compiler._component_trace.cost_query import (
    GATE_COST,
    PEAK_LOGICAL_QUBITS,
    SUCCESS_PROBABILITY,
    query_component_costs,
)

metrics = (GATE_COST, PEAK_LOGICAL_QUBITS, SUCCESS_PROBABILITY)
table = query_component_costs(model, metrics)
assert table[model][PEAK_LOGICAL_QUBITS] == 22
```

A component can declare `success_probability` to override its local
probability model. Without a declaration, the estimator multiplies child
probabilities, raised to their call multiplicities. A zero-call edge has no
effect. The probability override leaves gate and width aggregation intact.
This models declared algorithm success. An And uncompute, for example, can
have two equiprobable measurement histories that both restore the state and
therefore succeed with probability one.

Concrete call aggregation uses explicit sequential bystanders. The
shape-level formulas in
`flagquantum.compiler.registers.symbolic_resource_model` can also accept
SymPy expressions. The concrete path needs no SymPy import.

```python
from sympy import Symbol
from flagquantum.compiler.registers.symbolic_resource_model import (
    source_add_resource_formula,
    source_qrom_resource_formula,
)

n = Symbol("n", positive=True, integer=True)
adder = source_add_resource_formula(n, n)
lookup = source_qrom_resource_formula((n,), (2,), control_count=1)
assert adder.measurement.subs(n, 5) == 4
assert lookup.and_bloq.subs(n, 5) == 4
```

The symbolic Add width follows the source signature fallback. Rebuilding
the formula with concrete widths includes the decomposed carries. A QROM
formula uses the shape-only source model, including its product across
target word widths. Concrete table costs use the separate unary QROM model
and its early termination for uniform subtrees.

Shape-level blocked lookup and erasure formulas live in
`flagquantum.compiler.registers.symbolic_qroam_resource_model`.
They preserve the source's default and explicit block exponents. A
shape-only erasure returns separate wrapper and leaf widths and uses
the same original extent for both Toffoli formulas. Padding of the
erase leaf is specific to the concrete data model above. Bind concrete
sizes and rebuild the model to select integer blocks with the concrete
optimizer.

The shape-only rotation models in
`flagquantum.algorithms.primitives.symbolic_rotation_resource_model` reuse
the QROM, phase-gradient lookup and permutation-cycle formulas. The locked
dense call graph represents the amplitude layers with one full-shape lookup
and uses one more for the phase step. The sparse model uses one cycle of
length twice the declared number of nonzero coefficients. An explicit
target width affects the outer signature. The source still sizes the
unknown permutation from the state size.

The symbolic permutation width uses the locked callee fallback. With a
state size of 1,024, one nonzero coefficient and two phase bits, substituting
1,024 into the symbolic sparse width gives 19 logical qubits. Rebuilding at
the concrete width gives 20 because the equality has a decomposition at
that width. Both are shape models. Concrete coefficient data use the
separate rotation-tree and cycle construction.

Use `symbolic_call_graph` to apply the same traversal to declared symbolic
call multiplicities. Each terminal supplies the cost for one invocation.
For a terminal selected by a cutoff, provide the cost of that whole
component. Represent any exclusive interior costs as explicit terminal calls.

```python
from sympy import symbols

from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler.registers.symbolic_resource_model import (
    source_qrom_resource_formula,
)

repeats, rows = symbols("repeats rows", positive=True, integer=True)
graph = symbolic_call_graph("program", {"program": {"lookup": repeats}})
lookup = source_qrom_resource_formula((rows,), (8,), control_count=1)
t_count = graph.estimate_leaf_cost(lambda _: 4 * lookup.and_bloq)
measurements = graph.estimate_leaf_cost(lambda _: lookup.measurement)
assert t_count.subs({repeats: 3, rows: 17}) == 192
assert measurements.subs({repeats: 3, rows: 17}) == 48
```

This counts a shape-only program under the source gate model. Use
`estimate_connection_width` for sequential logical width when register
connections and each component's local peak are known. It schedules
ready steps in the source's stable greedy order, tracks bystander
registers, and takes the largest peak across weakly connected components.

```python
from flagquantum.compiler._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost

adder = source_add_cost(2, 2)
steps = [WidthStep("input"), WidthStep("component", adder.qubits), WidthStep("output")]
connections = [
    WidthConnection(0, 1, 2),
    WidthConnection(0, 1, 2),
    WidthConnection(1, 2, 2),
    WidthConnection(1, 2, 2),
    WidthConnection(0, 2, 3),
]
assert estimate_connection_width(steps, connections) == 8
```

The two 2-qubit operands and one carry belong to the adder's local peak.
The 3-qubit bystander remains live across the call. Bare `CircuitIR`
has no register ownership or release operations. For a verified
construction program, `estimate_register_program_width` derives
connections from declared registers, releases and ordered effects.
Outputs are reserved at entry, matching register lowering. Each repeated
operand reference counts as one register.

```python
from examples.synthesis_local_registers import local_composition
from flagquantum.compiler._construction.register_width_model import (
    estimate_register_program_width,
)

program, functions = local_composition(overlapping=True)
assert estimate_register_program_width(program, functions) == 8
```

The default gives the declared storage peak. A `local_peak` callback can
supply an expanded effect's modeled peak, including its distinct operands
and any workspace. The estimator keeps the other live registers as
bystanders and respects the original effect order.

Resource call keys preserve gate parameters for call-graph grouping. The
optional rules in `resource_generalization` can group rotations by family
and precision, group conjunctions by arity, and omit bookkeeping calls.
For comparison with the source's swap abstraction, the approximate swap
rule maps forward and inverse calls to the exact controlled-swap key.
Use unmodified call keys when counting approximate swap costs.

```python
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.resource_generalization import (
    ResourceCall,
    RotationCall,
    generalize_rotation,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt

root: ResourceCall = ComponentCost("program", 1)
calls: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {
    root: {RotationCall("rx", 0.2): 2, RotationCall("rx", -0.1): 3},
}
graph = symbolic_call_graph(root, calls, generalizer=generalize_rotation)
assert graph.leaf_counts == {RotationCall("rx", None): 5}
```

`classify_resource_gate` uses the source's finite angle and exponent
lists and its strict tolerance to classify explicitly modeled Clifford,
T-like and rotation calls. `ignore_known_cliffords` can remove the
recognized Clifford calls, including inside partition wrappers. Apply
that filter before generalizing angles if the classification needs the
original angle. An opaque component retains its own identity regardless
of its name.

Use `attribute_t_cost` for a model T budget by component family.
A recognized family stops graph expansion at that component. The
component's full unit cost is then multiplied by its weighted call
count. Unclassified terminal calls contribute to `other`. A component
that declares exclusive T costs on an expanded interior node should
expose those costs as child calls to include them in the attribution.

```python
from flagquantum.compiler._component_trace.resource_classification import attribute_t_cost
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall
from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost

adder = source_add_cost(4, 4)
program = ComponentCost("program", 11, calls=(CostCall(adder, 3),))
assert attribute_t_cost(program, lambda call: "arithmetic" if call == adder else None) == {
    "arithmetic": 36
}
```

`estimate_terminal_t_cost` reproduces the locked source's deprecated
terminal-count convention when comparing historical estimates. Each
recognized rotation contributes
`ceil(1.149 * log2(1 / precision) + 9.2)` before multiplication by its
call count. A bare T-like rotation angle contributes zero under this
convention. Use an explicit T key to count a T gate. The exact bitwise
controlled swap contributes seven T gates, whereas the default QEC
conversion assigns four T equivalents to the CSWAP category.

```python
from flagquantum.compiler._component_trace.resource_generalization import (
    FixedTCall,
    ResourceCall,
    RotationCall,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.terminal_t_cost import estimate_terminal_t_cost

sigma: dict[ResourceCall, SymbolicInt] = {
    RotationCall("rx", 0.2, precision=1e-4): 3,
    FixedTCall("fredkin"): 1,
}
assert estimate_terminal_t_cost(sigma) == 82
```

The single-control rule charges a rotation at nonintegral Pauli power
exponents. The locked source lacks a stabilizer verdict for symbolic
controlled angles and lacks a precision for controlled S or T. The
terminal model reports those cases as unsupported. Its totals are
separate from emitted gate counts.

`count_selected_calls` counts exact declared gate or component keys.
Each matched key stops expansion along that path. The preset
`t+tof+cswap` returns separate T, T adjoint, Toffoli and Fredkin keys.
Pass the leaf keys of a declared call graph to count at its leaf
boundary. Other explicit selections can include interior components.

```python
from flagquantum.compiler._component_trace.resource_generalization import (
    DeclaredResourceCalls,
    FixedTCall,
    PhaseCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.selected_call_count import (
    count_selected_calls,
    resource_gateset,
)

program = ComponentCost("program", 3)
tof = FixedTCall("toffoli")
t = PhaseCall("t")
tdg = PhaseCall("t", adjoint=True)
calls: DeclaredResourceCalls = {
    program: {tof: 2, tdg: 1},
    tof: {t: 4},
}
assert count_selected_calls(
    program, resource_gateset("t+tof"), declared_calls=calls
) == {tof: 2, tdg: 1}
assert count_selected_calls(
    program, resource_gateset("t"), declared_calls=calls
) == {t: 8, tdg: 1}
```

The gate budget converters accept the same categories from concrete
costs and symbolic formulas. T/CCZ accounting charges rotations to T
and preserves Toffoli-like operations as separate CCZ units. The
Toffoli-only view requires provably zero T and rotation counts.
The historical view keeps rotations separate and folds measurements
into the Clifford field. The Beverland view uses rotation count as
its `D_R` proxy. Actual rotation depth requires scheduling evidence.

```python
from flagquantum.compiler._component_trace.gate_budget import (
    beverland_gate_budget,
    legacy_gate_budget,
    t_and_ccz_budget,
)
from flagquantum.compiler._component_trace.resource_model import GateCost

gates = GateCost(t=2, toffoli=3, cswap=3, and_bloq=4, rotation=6, measurement=7)
assert t_and_ccz_budget(gates) == {"n_t": 68, "n_ccz": 10}
assert legacy_gate_budget(gates)["t"] == 51
assert beverland_gate_budget(gates)["D_R"] == 6
```

Use `query_qec_gate_cost` to compute these categories directly from
typed resource calls and symbolic multiplicities. Known leaves stop
expansion. Explicitly modeled zero-cost operations contribute zero,
and an unsupported key without declared callees raises an error.
Use a separate cache for each model configuration.

```python
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    FanoutCall,
)

assert query_qec_gate_cost(FanoutCall(12)).clifford == 1
assert query_qec_gate_cost(FanoutCall(12), legacy_shims=True).clifford == 23
erase = ConjunctionCall((0, 0), adjoint=True)
assert query_qec_gate_cost(erase).measurement == 1
assert query_qec_gate_cost(erase, legacy_shims=True).clifford == 7
```

Legacy annotations supply the historical T, Clifford and rotation
categories after seeded and static costs. Legacy fanout requires
a positive concrete width because the locked source uses its
decomposition for this convention.

At the optional Cirq boundary, `cirq_gate_resource_key` maps supported
gates to the same resource identities. Exact gate instances take
precedence over parameterized powers. For example, CZ maps to a
Clifford, while a fractional CZ power maps to a conjunction, a Z
power and measured uncompute. Unrecognized gates and unsupported
control wrappers return `None` at the adapter boundary.

```python
import cirq

from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.ecosystem.cirq.resource_keys import cirq_gate_resource_key

call = cirq_gate_resource_key(cirq.CZPowGate(exponent=0.3))
assert call is not None
cost = query_qec_gate_cost(call)
assert (cost.and_bloq, cost.rotation, cost.measurement, cost.clifford) == (1, 1, 1, 1)
```

Supported controlled gates with a single binary activation pattern
reuse the positive-controlled child. A negative single control
adds two X gates. Multiple controls count the predicate and its
measured inverse around the child. For a structured integer predicate,
encode the activation values with `ControlRegister.bits()` in port
order before constructing a `ControlPredicateCall`. A symbolic
control width can be supplied when the polarities are unspecified.

```python
call = cirq_gate_resource_key(cirq.ControlledGate(cirq.T, control_values=(0,)))
assert call is not None
cost = query_qec_gate_cost(call)
assert (cost.rotation, cost.clifford) == (1, 2)
```

The dedicated controlled Rz uses two Z powers and two CNOTs.
Non-singleton control conditions and wrappers over unsupported child
gates stay at the adapter boundary.

Adding a positive control to CNOT selects a Toffoli. Other CNOT
control patterns wrap that Toffoli in the predicate model. Adding
controls to Toffoli instead merges its two existing positive controls
into the new predicate. For a register CSwap, the source's additional
positive control decomposes one bit at a time, charging the predicate
once per bit. The bitwise Fredkin can merge its original control
directly with one added control.

The register emitters `controlled_rz_ir` and
`controlled_z_power_ir` share the same source decompositions as their
resource models. CRz takes radians and emits two half-angle phases
and two CNOTs. CZPow takes an exponent and emits a phase of
`pi * exponent` on the And result. Its clean work qubit and
measurement slot belong to the caller. Negating the parameter
emits the corresponding inverse.

```python
from flagquantum.compiler.registers.controlled_rotation import (
    controlled_rz_ir,
    controlled_z_power_ir,
)

crz = controlled_rz_ir(0, 1, angle=0.3)
czpow = controlled_z_power_ir(0, 1, exponent=0.3, work=2, classical_bit=0)
assert [op.name for op in crz] == ["phase", "cx", "phase", "cx"]
assert sum(op.name in {"t", "tdg"} for op in czpow) == 4
assert sum(op.name == "measure" for op in czpow) == 1
```

The emitted phase instructions retain their angles for later synthesis.
The QEC model classifies those angles separately as Clifford, T-like
or arbitrary rotations.

The native Pauli power emitters preserve the eigenphase at zero
global shift. For example, the square of X is the identity, whereas
a radian X rotation by twice pi is minus the identity. For a controlled
power, the X/Y global shift becomes a phase on the control wire. Qualtran
delegates these leaf unitaries to Cirq. FlagQuantum uses phase instructions
and basis changes to preserve the full complex action.

## X and Z basis states

`basis_state_ir` prepares a basis vector on a caller-owned clean
wire. The source's `|0>`, `|1>`, `|+>`, and `|->` emit, in order,
the empty sequence, X, H, and X followed by H. The clean wire binds
the source's single output port. Preparation introduces no additional
work wires or measurements.

`basis_state_cost_key` distinguishes the vectors and retains the
source's zero QEC gate cost. The X and H instructions that prepare
them remain visible in the emitted instruction count.

- [Qualtran v0.7.0 X basis](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py)
- [Qualtran v0.7.0 Z basis](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py)

## X and Z basis effects

The source's four LEFT effects consume one wire and have zero modeled
QEC gate cost. Its Cirq exporter refuses these effects. FlagQuantum's
local tensor-network path attaches the corresponding covector to the
selected output label and contracts it with the network.

`tensor_network_basis_effect` takes an existing contraction plan and
a map from wire to `(basis, bit)`. It returns the unnormalized surviving
ket, with the original batch axis and the unprojected wire order. A
zero-probability projection returns the zero ket. The contraction creates
no circuit instructions, auxiliary wires, or measurement samples.

## Integer basis vectors

`int_state_ir` binds an unsigned integer to caller-owned clean wires in
big-endian order. The source joins a sequence of zero and one states.
The emitted stream contains one X for each set bit. `int_effect_bindings`
binds the source's corresponding sequence of Z covectors to the existing
partial contraction. It preserves unnormalized complex amplitudes and
the surviving wire order. Neither path requires auxiliary qubits or
measurement samples.

The source declares one arbitrary Clifford for either vector, even when
preparation emits zero or several X gates. `int_vector_cost_key` preserves
this declaration for concrete and symbolic widths. The source models
width zero, but its state and effect decompositions both fail at that
width. FlagQuantum requires a nonempty physical binding. The controlled
modular constant adder shares the load stream. On its declared input
domain the adder preserves the loaded word, so the matching unload
returns the allocated work wires to zero. Its existing source call graph
charges the controlled modular add alone.

## Consuming X and Z measurements

The pinned `MeasureX` and `MeasureZ` each consume one quantum wire and
produce one classical bit. The model charges one measurement and one
peak logical qubit per leaf. In native dynamic IR, the caller supplies
the wire and the classical result slot. Z uses a computational-basis
measurement followed by X on outcome one to clear the consumed physical
slot for reuse. X prepends H to the same sequence. The emitted H and
conditional X are reported separately from the source cost. The output
slot retains the measured bit after the physical wire is cleared.

For exact tensor-network analysis,
`tensor_network_basis_measurement(plan, wire, basis=...)` returns the
unnormalized survivor ket for classical outcomes 0 and 1 in that order.
It removes the measured quantum axis and keeps the batch axis and the
original order of all spectator wires. The squared norm of each ket is
its outcome probability, including zero for an impossible outcome.
Taking the outer product of each branch in a separate classical block
reproduces the source superoperator.

## Discarding quantum and classical outputs

`discard_basis_measurement_result` traces the classical result of an X
or Z measurement. It sums branch density matrices and preserves each
branch's probability weight. `tensor_network_discard_qubit` uses the
same partial trace to consume one quantum output. Both keep spectator
wires in their original order. Tracing out the only wire returns a
batched one-by-one matrix containing the input trace.

The pinned `Discard` consumes a classical bit with zero modeled gates
and zero quantum width. Its tensor behavior agrees with tracing the
classical result of a composed measurement. The pinned `DiscardQ`
consumes a quantum wire, reports peak width one, and has no QEC gate
cost declaration. Its composed quantum behavior agrees with the
native partial trace. On fixed-width dynamic IR, `discard_qubit_ir`
emits one reset so the consumed physical slot is reusable in zero.
It allocates no classical result bit. That physical reset is an
emitted instruction and is reported separately from the source model.

## Register identity

The source models Identity as one atomic call with zero QEC gates.
Quantum controls widen the identity by the control register widths.
They require no predicate, work qubits, or measurements. The native
`register_identity_ir` expands the atomic call to an explicit `i`
on each bound wire, in control-group order followed by target order.
A zero-width binding emits no instructions.

`identity_cost_key` retains the concrete or symbolic total width.
Cirq's resource adapter uses this same key for recognized single-wire
identity and controlled identity wrappers. A direct multi-wire Cirq
identity stays at the adapter boundary, matching the locked mapper.

- [Qualtran v0.7.0 Identity](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/identity.py)

## Applying a leaf to each wire

`on_each_ir` binds an unconditioned single-qubit gate to each wire in
the supplied register order. Split and join are wire bookkeeping.
An empty register emits no gates. The same path emits the existing
`bitwise_not_ir` X sequence, whose register must be nonempty.

`OnEachCall` contributes one child call per wire. Its width may be
symbolic in the cost model. For atomic one-qubit leaves the locked
source models a peak of one qubit per wire. A concrete wire binding
is required for emission.

- [Qualtran v0.7.0 OnEach](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/on_each.py)

## Repeated register operations

`register_power_ir` repeats a positive concrete number of calls to a
bound in-place operation, in circuit order. Its output preserves literal
rotation calls, including repeated calls to a single rotation. A body
that uses classical feedback must define all inputs to that feedback
within each iteration. The last iteration's writes remain in the
caller-provided classical slots. For measured adjoints, supply the
operation's explicit inverse as the body.

`PowerCall` contributes one child edge with a concrete or symbolic
multiplicity to the resource graph. It multiplies gate counts without
materializing instructions. The locked source gate model supports
symbolic counts, while its width query raises when it attempts to
decompose a symbolic power. Concrete executions reuse the same
caller-owned scratch and do not increase width with the repetition count.

- [Qualtran v0.7.0 Power](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/power.py)

## Controlled Hadamard

The native `multi_control_h_ir` emitter uses H without controls and
the atomic CH instruction for one positive control. A negative
control wraps CH in two X gates. Multiple controls compute the source
AND predicate, apply CH to the target, and erase the predicate with
measurements and feedback. Each control beyond the first needs one
caller-owned clean work qubit and one classical slot.

The locked source assigns each CH four Clifford gates and two rotations
in the QEC model. FlagQuantum emits one CH instruction, leaving basis
lowering to the target compiler. For a two-control pattern, the native
predicate adds four emitted T-family instructions, one measurement and
one work qubit. The resource model accounts for the predicate and CH
separately.

- [Qualtran v0.7.0 Hadamard](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/hadamard.py)

## Fixed Pauli controls

`pauli_gate_ir` binds a fixed X, Y or Z to an ordered group of binary
controls. A single positive control emits CX, CY or CZ. The source
specializes the two-wire all-positive X control group to atomic CCX
with no work or measurement. Other multi-wire groups use the shared
AND predicate and measured cleanup. A single negative control wraps
the positive leaf with X before and after.

The compiler and the Cirq adapter use the same QEC keys. The atomic CCX
is one modeled Toffoli. Its emitted instruction has no explicit T gates
before basis lowering. The general `multi_control_x_ir` ladder retains
its original behavior for callers that request that decomposition.

- [Qualtran v0.7.0 X basis](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/x_basis.py)
- [Qualtran v0.7.0 Y gate](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/y_gate.py)
- [Qualtran v0.7.0 Z basis](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/z_basis.py)

## Additional controls on CY and CZ

`controlled_pauli_with_controls_ir` preserves the source's nested
predicate schedule. A single added control enters a binary AND before
the original positive control. For several added controls, an outer
AND reduces them to one wire. The inner binary AND combines that result
with the original control, then the native CY or CZ acts on the target.
Measurement feedback clears the inner predicate before the outer one.

With two added controls, the emitted wrapper uses eight T-family
instructions, two clean work wires, and two measurements. The locked
QEC model counts two And calls, three Clifford gates, and two
measurements at width six. For CZ, the emitted stream includes the
unconditional CZ leaf and two conditional CZ corrections.

- [Qualtran v0.7.0 specialized control](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py)

## Additional controls on CNOT and Toffoli

`cnot_with_controls_ir` binds one positive added control to a native
CCX in source operand order: new control, original control, target.
A negative control wraps the CCX in two X instructions. With two or more
added controls, the predicate includes only the added controls and the
result controls the same CCX leaf.

`toffoli_with_controls_ir` follows the source's other route. It puts
the added controls before the two original positive controls in one
predicate, applies CX from the predicate to the target, then clears
the predicate with measurement feedback. For one added control the
native wrapper emits eight T-family instructions and two measurements
with two work wires. The QEC model counts two And calls and the CX
and cleanup separately.

- [Qualtran v0.7.0 CNOT](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/cnot.py)
- [Qualtran v0.7.0 Toffoli](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/toffoli.py)

## Fixed S and T phases

`phase_family_ir` emits S, S†, T or T† directly. Under a positive
control it binds the source's atomic controlled phase to a native
`cphase` with the corresponding signed angle. The source does not
decompose this leaf. Its QEC model counts one rotation, while the
precision-sensitive terminal model cannot estimate its T cost because
the controlled leaf does not supply a precision.

A single negative control wraps the leaf in two X instructions.
Multiple controls reuse the measured AND predicate and its inverse.
For two controls the wrapper emits four T-family instructions, uses
one clean work wire and measures once. These counts exclude subsequent
basis synthesis of the controlled phase.

- [Qualtran v0.7.0 S gate](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/s_gate.py)
- [Qualtran v0.7.0 T gate](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/t_gate.py)

## Scalar and controlled phases

Qualtran's atomic global phase multiplies amplitudes by
`exp(i * pi * exponent)` and has zero modeled gates and zero logical
ports. FlagQuantum emits two X and two phase instructions on one
existing wire to retain that scalar in a circuit. The carrier can be
entangled and requires no initialization.

A positive single control uses one Z power. A negative control
wraps it in two X gates. For multiple controls, FlagQuantum follows
the source's predicate, Z power and measured inverse. Callers bind
the controls and activation values in matching order and provide
one clean qubit and one classical slot for each control beyond the first.

```python
from flagquantum.compiler._component_trace.compound_control_resource import (
    global_phase_cost_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.global_phase import global_phase_ir

gates = global_phase_ir(
    0.25, controls=(0, 1), values=(0, 1), workspace=(2,), classical_bits=(0,)
)
assert sum(gate.name in {"t", "tdg"} for gate in gates) == 4
assert sum(gate.name == "measure" for gate in gates) == 1
estimate = query_qec_gate_cost(global_phase_cost_key(0.25, (0, 1)))
assert (estimate.t, estimate.and_bloq, estimate.measurement) == (1, 1, 1)
```

The emitted phase instruction retains its angle. The model recognizes
that angle as one additional T independently of the four emitted T-family
instructions in the AND computation.

Use `global_phase_exponent` to obtain the principal argument of a
complex coefficient in units of pi. As in the source, the coefficient
magnitude is discarded and a concrete zero maps to the identity phase.

- [Qualtran v0.7.0 global phase](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/global_phase.py)

## SU(2) rotation decomposition

`su2_rotation_ir` emits the source's scalar phase followed by Rz, Rx
and Rz in circuit order. Negating the global shift and exchanging and
negating the two phase parameters gives the source adjoint. All three
rotations and the scalar use the target wire, with zero additional
qubits and zero measurements.

`su2_rotation_calls` preserves the source's symbolic parameters and
divides the requested synthesis precision by three for the three
rotations. The scalar keeps the source's default precision and zero
modeled cost. The emitted circuit retains the three rotation angles
for later synthesis.

Use `su2_angles_from_matrix` to recover the four parameters from a
2-by-2 unitary. The source sets one phase to zero when the sine or
cosine is within an absolute tolerance of 1e-8. For a near-singular
input, the reconstructed unitary can therefore differ slightly from
the input. Use `su2_angles_from_zxz` to convert first Rz, middle Rx,
and final Rz angles in circuit order, with an optional scalar shift.
The ZXZ conversion accepts symbolic angles for resource queries.

- [Qualtran v0.7.0 SU(2) rotation](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/su2_rotation.py)

## THC SELECT before lowering

For THC SELECT, the model needs only the dimensions, angle precision,
alias-sampling keep width and optional erasure block sizes. The block
sizes `kr1` and `kr2` default to one. No angle tables are required.

```python
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

model = select_thc_cost(num_mu=2, num_spin_orb=4, num_bits_theta=8, keep_bitsize=1)
estimate = estimate_component_resources(model)
assert estimate.gates.toffoli == 107
assert estimate.gates.cswap == 11
assert estimate.gates.total_t_count() == 472
assert estimate.peak_logical_qubits == 24
```

The 24 logical qubits include the source's keep register, rotation flag,
shared angle register and one temporary bit for the two-control reflection.
The source declares zero measurements because its THC rotations are
cost-only leaves. To report a FlagQuantum implementation alongside this
model, use `compare_component_resources(model, circuit.to_ir())` after
`circuit.append(op)`. For the two-index, four-spin-orbital test case, the
8-bit native path with measurement erasure and the measured adder declares
39 wires (13 data and 26 work). It emits 112 T/T†, 60 measurements, 49
CCX and 11 CSwap instructions, among other gates. The remaining compound
gates still need synthesis. The 472 T equivalents are the logical model's
conversion at four per Toffoli or CSwap.

- [Qualtran v0.7.0 THC SELECT and rotation costs](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/thc/select_bloq.py)
- [Qualtran v0.7.0 controlled reflection costs](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/black_boxes.py)

- [Qualtran v0.7.0 gate costs](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py)
- [Qualtran v0.7.0 logical width](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py)
- [Qualtran v0.7.0 success probability](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_success_prob.py)
- [Qualtran v0.7.0 call graph](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_call_graph.py)
- [Qualtran v0.7.0 cost queries](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_costing.py)
- [Qualtran v0.7.0 symbolic mathematics](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/math_funcs.py)
