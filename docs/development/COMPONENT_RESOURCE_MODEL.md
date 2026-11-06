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

## Unified FTQC resource report

Call the experimental `estimate_ftqc_resources` entry point in
`flagquantum.qec.resources.estimation` with a declared component model. This
works before circuit construction. Pass `emitted_ir=circuit.to_ir()` to include
instruction counts and both `error_budget` and `physical_error` to include
the Beverland fast data-block estimate.

Operations with a zero-argument `resource_model()` returning a concrete
`ComponentCost` can be passed directly. For example, plan a bounded Pauli
SELECT, then count its emitted instructions after binding it to a circuit.

```python
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import SelectPauliLCU
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

op = SelectPauliLCU(2, 1, ("X", "Z", "X")).make_bounded_on(
    selection=(0, 1), target=(2,), workspace=(3,), classical_bits=(0,)
)
plan = estimate_ftqc_resources(op)
assert plan.logical.gates.total_t_count() == 4
assert plan.logical.peak_logical_qubits == 4

circuit = DynamicCircuit(4).append(op)
report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
assert report.emitted is not None
assert report.emitted.t_count == 4
assert report.emitted.unconditional["measure"] == 1
```

For a direction or composition that needs model arguments, evaluate its
model with those arguments and pass the result. Shape formulas with
unresolved symbols use the separate symbolic resource queries.

```python
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

model = select_thc_cost(num_mu=2, num_spin_orb=4, num_bits_theta=8, keep_bitsize=1)
report = estimate_ftqc_resources(model, error_budget=1e-3, physical_error=1e-4)
assert report.logical.gates.total_t_count() == 472
assert report.logical.peak_logical_qubits == 24
assert report.surface_code is not None
assert report.surface_code.data_block_physical_qubits == 6174
```

Run `python -m examples.algorithms.ftqc_resource_estimation` to also construct
the 8-bit circuit with `circuit.append(op)`. The emitted section reports 112
T/T† instructions, 49 CCX instructions, 11 CSwap instructions and 60
measurements across 39 declared wires. These are static counts at the emitted
IR level, before further synthesis or measurement branch selection. The
surface-code scenario uses the declared model and is independent of the
optional circuit. The caller pairs the model with its corresponding circuit.

## Gidney-Fowler data and factory scenario

Pass `report.logical` to
`flagquantum.qec.resources.gidney_fowler.estimate_gidney_fowler_resources`.
The scenario uses the locked Gidney-Fowler logical error model, a simple data
block with 50% routing overhead, and one catalyzed CCZ-to-T factory. Supply the
data code distance and optionally the two factory distances, physical error
rate, and cycle time. Each rotation contributes eleven T states in the
source's gate budget. The simple data block assumes unlimited magic-state
consumption, so the factory sets the cycle count.

At data distance 7, physical error rate 0.0001, and one microsecond per cycle,
the 8-bit THC example takes 20,119 cycles (20.119 ms) in the model. The simple
data block uses 3,528 physical qubits and the factory uses 147,904. The
first-order failure estimate is 0.000724284. This scenario is computed
independently of the Beverland data-block calculation and the emitted IR
counts above.

Set `data_layout` to `"compact"`, `"intermediate"`, or `"fast"` to
model consumption at nine, five, or one surface-code step per magic
state. Set `factory_count` for parallel identical factories. Their
footprint scales with the count and production cycles divide by the count,
rounded up. Total cycles are the larger of production and consumption.
For the same THC workload, a distance-21 compact block with one factory
takes 22,302 cycles, limited by the data block. The default simple layout
and one factory retain the original 20,119-cycle estimate.

## 15-to-1 distillation factory estimate

`FifteenToOneFactory` in `flagquantum.qec.resources.fifteen_to_one` emits
the five-wire distillation schedule as FlagQuantum IR. Wire 0 carries the
output. The estimate postselects the four check wires on the all-plus
state using the native density-matrix executor. It reports the expected
cycles and output error for a supplied T-state demand along with the
factory footprint. The logical error scaler is 0.1 for the Gidney-Fowler
scheme and 0.03 for the Beverland scheme, both with threshold 0.01.

For distances (7, 3, 3), physical error 0.0001, and scaler 0.1, one
requested T state takes 19 modeled cycles and the factory occupies 810
physical qubits. The conditional output error is about 4.4e-8. The
density matrix and resource values were compared with the locked source.

## Beverland architecture with a 15-to-1 factory

Pass the same logical report to
`flagquantum.qec.resources.beverland_physical.estimate_beverland_physical_resources`.
The source preset combines a compact block, a (9, 3, 3) distillation factory,
a 0.001 physical error rate, and a 0.4 microsecond superconducting cycle.
Choose the data distance and optionally override the layout, factory,
hardware, or number of identical factories. The factory budget converts
logical gates to T equivalents. The data block uses the source's T and CCZ
consumption budget.

The 8-bit THC example at data distance 25 yields 46,146 physical qubits
and 26,550 cycles (10.62 ms), limited by the compact layout's consumption
rate. Run `python -m examples.algorithms.ftqc_resource_estimation` to
inspect this architecture, the data-block-only estimate, and the emitted
instructions separately.

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

Use `estimate_symbolic_ftqc_resources` to report the logical costs of an
existing shape formula before supplying a table or constructing a circuit.
The report keeps the gate categories and peak logical width, and gives both
the default T-equivalent cost and the separate T and CCZ state budgets.

```python
from sympy import symbols

from flagquantum.compiler.registers.symbolic_resource_model import (
    source_qrom_resource_formula,
)
from flagquantum.qec.resources.estimation import estimate_symbolic_ftqc_resources

rows = symbols("rows", positive=True, integer=True)
formula = source_qrom_resource_formula((rows,), (8,), control_count=1)
estimate = estimate_symbolic_ftqc_resources(formula)
assert estimate.t_equivalents.subs(rows, 17) == 64
assert estimate.ccz_states.subs(rows, 17) == 16
assert estimate.logical.measurement.subs(rows, 17) == 16
assert estimate.logical.logical_qubits.subs(rows, 17) == 19
```

These values describe the shape-only model. For a concrete component, use
`estimate_ftqc_resources` and supply its emitted IR to count the instructions
in the chosen lowering.

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

## Allocate and free workspace

The source's Allocate and Free bookkeeping leaves each have zero modeled
gates and a peak logical width equal to the register size. FlagQuantum's
`RegisterWorkspace` binds caller-owned wires in order and returns them
to their original clean or borrowed pool. Freeing a whole-register lease
makes those wires available for reuse. It emits no reset or quantum
gates. The polynomial and finite-field emitters use this pool and keep
their existing ordered allocation and scratch records.

A clean lease must begin in zero and return to zero before release.
A borrowed lease can carry arbitrary amplitudes and spectator
entanglement. Its consumer must restore the joint state before release.
The pinned Cirq adapter distinguishes clean allocation from borrowing.
Its tensor Allocate/Free view uses zero even with the dirty flag.
Use the zero covector to compare that tensor view, and check an actual
borrowed circuit with a coherent input to verify restoration.

## Split and join register views

`split_register` binds an ordered word to one-wire ports, and
`join_register` packs those ports in the supplied order. Reordering the
ports changes the logical wire mapping without emitting SWAP gates.
The source's concrete leaves have zero modeled gates and peak logical
width equal to the original word width. Both accept an empty physical
binding. The pinned leaves reject symbolic dtypes.

Integer values use a separate big-endian encoder and decoder. Signed
words use two's complement. Integer effects share the split binding and
unsigned encoder. Integer conversion requires a positive concrete width.
The pinned source accepts a zero-width physical binding, while its
zero-width integer conversion fails.

## Named partitions and two-word views

`partition_register` and `unpartition_register` bind a flat word to
named ports and back. A shaped port is flattened in row-major order.
The inverse takes ports in declared layout order, as the source tensor
and classical compositions do. Calling the source Cirq adapter
directly with reordered keywords can give a different order.
`split_register_pair` and `join_register_pair` use the same layout rule
for a high word and a low word. The sparse rotation preparation uses
the low-word binding directly.

These bindings emit no gates. The source model has zero gate costs and
peak logical width equal to the sum of port widths, including symbolic
widths. Physical binding requires concrete widths and permits empty
ports. The source classical Split2 rejects empty integer outputs.
Classical Join2 accepts one empty input and rejects an empty result.
Its array conversion masks nonnegative machine words to the declared
width and checks the range of wider values.

## Cast register views

`RegisterCast` reinterprets a word or a shaped array of words at the
same width. It binds the same ordered slots and emits no gates.
`reinterpret_integer` encodes and decodes one stored word. A fixed-point
word is already a stored integer, so the cast applies no scale or
rounding. The source reports zero gate cost and the greater of its
input and output quantum widths, including any symbolic shape.

Casting between quantum and classical kinds requires explicit opt-in.
The cast is metadata only. Use a measurement to obtain a classical
outcome from a quantum state. The native adjoint retains the shape
and opt-in. The pinned source adjoint drops both, and its shaped
tensor export fails. The native shaped view uses the original Cirq
wire order and reports the pinned shape-level width.

## Always-executed child operations

`Always` passes inherited control bindings through and emits the wrapped
child with its original signature. It uses the child's declared inverse
when lowering the adjoint. In controlled AddK, only the constant load
and unload take the external control. The middle Add and its measurement
feedback execute on both control branches. The loaded word is zero on
the inactive branch, so the full operation still preserves the target
there and returns all work wires to zero.

The wrapper adds no gates, measurements or work wires.
`always_component_cost` retains the child's direct cost declaration and
callees. The pinned source forwards the decomposable child's graph.
An atomic source child without a call graph, such as X under Always,
has no source QEC cost for the wrapper. A native cost with an explicit
declaration remains usable.

## Opaque Clifford resource leaves

`arbitrary_clifford_cost_key` declares an abstract Clifford and preserves
its concrete or symbolic register width. The pinned leaf reports exactly
one Clifford even at zero width. The adjoint has the same cost and width.
The source supplies neither a gate decomposition nor a tensor for this
leaf. Executable consumers retain their own decompositions.

Integer vectors share this resource declaration. The approximate
controlled swap uses it to retain distinct one- and two-qubit Clifford
calls in its forward and inverse resource rows. For a swap of two
four-qubit registers the model reports 16 T-family gates and 87 abstract
Cliffords. The original primitive instruction count is tracked separately.

## Automatically partitioned ports

`AutoPartition` binds packed groups to named child ports in their declared
order. An unused segment passes through and rejoins its original group.
Repacking uses the child's actual output wire mapping, so virtual
permutations remain bookkeeping. `left_only` exposes the named child
outputs directly and supports consumed input ports. It rejects unused
segments, matching the pinned constructor. The pinned version also
rejects child signatures with right-only ports.

Partitioning emits no instructions and adds no scratch. Its resource
width includes unused wires live during the child. Symbolic widths are
available to the resource model, while binding requires concrete widths.
BlackBoxPrepare shares the same binding path.

## Literal register swaps

`swap_bit_ir` emits the atomic SWAP. `swap_register_ir` emits one SWAP
per corresponding bit pair in declared order. The operation preserves
its physical output bindings and is its own inverse. Field output
materialization uses the same leaf after choosing which wires must
be exchanged. A virtual output permutation alone emits no gates.

The pinned source models a Clifford per bit pair and twice the word
width in logical qubits. This formula accepts symbolic widths. The
native emitter requires a nonempty concrete pair, since the pinned
zero-width source fails during decomposition. The source's single
positive control specializes to the separately verified exact Fredkin
path. Its actual Clifford+T stream and abstract model are accounted
for separately from literal SWAP.

For a negative control or a mixed control register, use
`controlled_swap_with_predicate_ir`. Its control wires and activation values
follow the declared register order. A single negative control conjugates the
Fredkin stream with X on the control. Multiple controls share one prefix
conjunction per extra control across the entire register swap. Each prefix
uses one clean work qubit and one classical slot. The reverse measurement
sequence clears every work qubit and preserves the controls and their
relative phases.

For three-bit words and activation pattern `(1, 0, 1)`, the model reports
three CSwap, two And, six Clifford and two measurements with peak width 11.
At the source's default CSwap conversion this is 20 T equivalents. With
`ts_per_cswap=7` it is 29, matching the T/T† count emitted with exact Fredkin
leaves. The emitted instructions and the model retain separate Clifford
counts.

- [Qualtran v0.7.0 SWAP controls](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py)

### Outer controls on register CSwap

`control_register_cswap_ir` accepts an existing positive control and a group
of outer controls. It preserves the pinned source's per-pair sequence:
compute the conjunction of the outer predicate and existing control, emit
one exact Fredkin, then measure and erase that conjunction. Each pair reuses
the last work qubit and classical slot. Earlier work qubits and slots belong
to the outer predicate. For a single negative outer control, X conjugates
the full register block.

For three-bit words and outer pattern `(1, 0, 1)`, the source model reports
three CSwap, five And, nine Clifford and five measurements, with 13 peak
logical qubits. That is 32 default T equivalents or 41 with exact seven-T
Fredkin leaves. The emitted circuit uses 41 T/T†, five measurements and
three work qubits.

### Controls added to atomic Fredkin

`control_bit_cswap_ir` prepends a single outer control to the existing
activation pattern. For an outer zero and the original positive control,
the source calls And with polarities `(0, 1)` directly. Additional
single controls extend the same prefix ladder. The original positive
control remains the final entry in the activation pattern.

A group of multiple controls on atomic Fredkin uses an outer predicate
and an inner positive-control And. For outer pattern `(1, 0, 1)`, it
emits 19 T/T† and three measurements, uses three clean work qubits,
and returns them to zero in every branch. The source reports one
CSwap, three And, seven Clifford and three measurements, with peak
width nine. The default conversion is 16 T equivalents. The exact
Fredkin conversion is 19.

Adding multiple controls to an already merged atomic Fredkin raises
inside the pinned source's `numpy.concatenate` call. The native emitter
rejects that combination explicitly.

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

## First-quantized SELECT

`source_select_first_quantization_cost` models four momentum exchanges,
kinetic SELECT and potential SELECT. The pinned width schedule runs the
kinetic call before allocating the second momentum register. Both momentum
registers and the nuclear position register are live during potential SELECT.
The nuclear position register is released before the return exchanges.

```python
from flagquantum.algorithms.first_quantization_resource_model import (
    source_select_first_quantization_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

report = estimate_ftqc_resources(source_select_first_quantization_cost(2, 3, 2))
assert report.logical.peak_logical_qubits == 81
assert report.logical.gates.total_t_count() == 1184
assert report.logical.gates.total_t_count(ts_per_cswap=7) == 1400
```

The example uses two momentum bits, three particles, two atoms, the source
default nuclear precision of 16 bits, and a sampling range of 256. Its gate
categories are 208 Toffoli, 72 CSwap, 16 And, 82 Clifford and 16 measurements.
The pinned kinetic and potential SELECT leaves declare costs only. The
one-bit momentum source has a gate-count result, but its width query fails
on the signed-conversion signature. The combined estimator requires at least
two momentum bits.

- [Qualtran v0.7.0 first-quantized SELECT call graph](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py)

## First-quantized kinetic PREPARE

The same estimator accepts first-quantized PREPARE models before a circuit is
built. The pinned source declares 13 Toffolis for the coordinate register
`w`, while its composite uses uniform preparation over three values.
Select the composite cost explicitly to see that difference.

```python
from flagquantum.algorithms.first_quantization_resource_model import (
    source_prepare_t_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

declared = estimate_ftqc_resources(source_prepare_t_cost(6))
composite = estimate_ftqc_resources(source_prepare_t_cost(6, decomposed=True))
assert declared.logical.gates.toffoli == 21
assert composite.logical.gates.toffoli == 8
assert composite.logical.gates.measurement == 5
assert declared.logical.peak_logical_qubits == 18
assert composite.logical.peak_logical_qubits == 18
```

For six momentum bits, the resource counts are:

| Scope | Logical Toffoli | Logical And | Logical Clifford | Logical rotations | Logical measurements | Peak logical qubits |
|---|---:|---:|---:|---:|---:|---:|
| Declared PREPARE T | 21 | 0 | 0 | 0 | 0 | 18 |
| Composite PREPARE T | 8 | 5 | 35 | 2 | 5 | 18 |

The emitted `w` preparation alone has 20 T/T† instructions, five
measurements, two Rz instructions and six declared wires, including four
clean work wires. Both PREPARE T estimates include the declared costs for
`r` and `s`. Their pinned `PreparePowerTwoState` has only a cost model,
as does the separate distinct-particle `i/j` preparation. The original
one-bit power-of-two cost of −1 Toffoli is preserved for source comparison.

## First-quantized momentum-transfer PREPARE

The momentum-transfer model retains the source's asymmetric forward and
adjoint resource declarations. Its optional composite cost adds the uniform
preparation of `m`, which is absent from the declared gate graph. Peak width
follows the source composite in both cases, including the three flags that
the forward inequality consumes and the adjoint inequality produces.

```python
from flagquantum.algorithms.first_quantization_nu_resource_model import (
    source_prepare_nu_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

forward = estimate_ftqc_resources(source_prepare_nu_cost(6, 32768))
adjoint = estimate_ftqc_resources(source_prepare_nu_cost(6, 32768, adjoint=True))
assert forward.logical.gates.toffoli == 567
assert forward.logical.peak_logical_qubits == 73
assert adjoint.logical.gates.toffoli == 20
assert adjoint.logical.peak_logical_qubits == 46
```

For these parameters the composite adds 15 modeled Clifford operations
in each direction. The shared uniform emitter implements the `m` subcall.
The remaining five source children supply only resource declarations, so
the whole preparation has no emitted instruction report. The pinned
composite rejects `m_param=1` because the zero-width uniform preparation
has no target port. The estimate raises for this case.

## First-quantized potential PREPARE

The potential preparation combines the momentum-transfer graph with the
nuclear-index model. The two register groups are independent. The nuclear
index remains live during momentum preparation, and vice versa. The
adjoint reverses the calls and follows the source's separate cost rules.

```python
from flagquantum.algorithms.first_quantization_resource_model import (
    source_prepare_uv_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

forward = estimate_ftqc_resources(source_prepare_uv_cost(6, 10, 32768, 10))
reverse = estimate_ftqc_resources(
    source_prepare_uv_cost(6, 10, 32768, 10, adjoint=True)
)
assert forward.logical.gates.toffoli == 577
assert forward.logical.peak_logical_qubits == 77
assert reverse.logical.gates.toffoli == 24
assert reverse.logical.peak_logical_qubits == 50
```

The locked nuclear-index leaf declares one Toffoli per unit of total nuclear
charge forward and the ceiling of the floating-point square root of that
charge for the adjoint. It has no gate decomposition. Its index is
zero-width for one atom, and the source width model still includes the
three-wire Toffoli callee. The example above reports analytical resources
for the potential preparation. Its complete emitted gate count is
unavailable.

## First-quantized top-level PREPARE

The top-level model combines term selection, distinct-particle preparation,
and kinetic and potential preparation. The three momentum-transfer axes
keep the source x/y/z order. The `plus_t` port is a bystander at each call.

```python
from flagquantum.algorithms.first_quantization_resource_model import (
    source_prepare_first_quantization_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

forward = estimate_ftqc_resources(
    source_prepare_first_quantization_cost(6, 10, 10, 10, m_param=32768)
)
reverse = estimate_ftqc_resources(
    source_prepare_first_quantization_cost(6, 10, 10, 10, m_param=32768, adjoint=True)
)
assert forward.logical.gates.toffoli == 680
assert forward.logical.gates.and_bloq == 28
assert forward.logical.peak_logical_qubits == 102
assert reverse.logical.gates.toffoli == 127
assert reverse.logical.peak_logical_qubits == 76
```

The report follows the locked source cost declarations. The top-level
graph contains leaves without gate decompositions, including term selection.
The source composite also requires at least two particles and a nonempty
`m` register. The model reports those unsupported boundaries explicitly.

## First-quantized projectile kinetic PREPARE and SELECT

The projectile resource model tracks separate electron and projectile
momentum widths. The source kinetic PREPARE uses two-bit coordinate words
and has different costs in each direction. The kinetic SELECT uses
three-bit coordinate words. The potential SELECT reuses the signed
conversion, addition and nuclear-phase models.

```python
from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    source_projectile_prepare_t_cost,
    source_projectile_select_uv_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

prepare = estimate_ftqc_resources(source_projectile_prepare_t_cost(6, 8))
reverse = estimate_ftqc_resources(source_projectile_prepare_t_cost(6, 8, adjoint=True))
potential = estimate_ftqc_resources(source_projectile_select_uv_cost(6, 8, 10, 16))
assert prepare.logical.gates.toffoli == 47
assert reverse.logical.gates.toffoli == 27
assert prepare.logical.peak_logical_qubits == 20
assert potential.logical.gates.toffoli == 669
assert potential.logical.gates.and_bloq == 42
assert potential.logical.peak_logical_qubits == 91
```

The locked projectile leaves declare resource costs without gate
decompositions. Kinetic PREPARE requires the projectile word to be at
least as wide as the electron word. Potential SELECT requires both words
to have at least two bits for the source signed-conversion signature.

## First-quantized projectile momentum-transfer and potential PREPARE

The projectile momentum-transfer model reuses the shared Nu graph and
substitutes the source projectile unary preparation. The forward potential
preparation combines that graph with the disjoint nuclear-index register.
The source still declares the unary flag as an output in its adjoint. The
adjoint composite cannot bind that flag, so the adjoint cost function
returns declared gate counts without a peak-width estimate.

```python
from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    source_projectile_prepare_uv_adjoint_gate_cost,
    source_projectile_prepare_uv_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

forward = estimate_ftqc_resources(
    source_projectile_prepare_uv_cost(6, 8, 10, 524288, 10)
)
reverse_gates = source_projectile_prepare_uv_adjoint_gate_cost(6, 8, 10, 10)
assert forward.logical.gates.toffoli == 941
assert forward.logical.gates.and_bloq == 36
assert forward.logical.peak_logical_qubits == 97
assert reverse_gates.toffoli == 33
```

The uniform `m` preparation uses the shared executable implementation.
The complete source graph includes model-only leaves, so these estimates
describe analytical resources.

## Controlled three-coordinate projectile exchange

The controlled exchange takes the leading electron-width bits of each
big-endian projectile word in x/y/z order. The remaining bits pass through,
including any entanglement. The source uses `cvs` as diagram labels and
its child activates on all-one controls. This behavior applies even when
the label is zero.

```python
from flagquantum.algorithms.first_quantization import (
    ControlledMultiplexedCSwap3D,
    source_controlled_multiplexed_cswap_3d_cost,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

op = ControlledMultiplexedCSwap3D.make_on(
    controls=(0,),
    cvs=(0,),
    selection=(1,),
    targets=(((2,), (3,), (4,)), ((5,), (6,), (7,))),
    junk=((8, 9), (10, 11), (12, 13)),
    workspace=(14,),
    classical_bits=(0,),
)
circuit = DynamicCircuit(15)
circuit.append(op)
report = estimate_ftqc_resources(
    source_controlled_multiplexed_cswap_3d_cost(2, 1, 2, 1),
    emitted_ir=circuit.to_ir(),
)
assert report.logical.peak_logical_qubits == 15
assert report.logical.gates.total_t_count() == 28
assert report.emitted is not None
assert report.emitted.t_count == 46
assert report.emitted.unconditional["measure"] == 1
```

The default estimate prices six abstract controlled swaps at four T
equivalents each. The emitted exchange uses seven T/T† instructions per
exact Fredkin and four for the measured And. The model and emitted width
both include three untouched suffix wires.

## First-quantized projectile top-level PREPARE and SELECT

The top-level SELECT model retains the source allocation schedule. It
accounts for the projectile-sized momentum words, the electron-sized
momentum words, and the nuclear-position register at their actual points
of use. Each projectile word leaves the schedule after its last swap.
The two addressed controlled exchanges follow the source activation
described above.

```python
from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    source_projectile_prepare_first_quantization_adjoint_gate_cost,
    source_projectile_prepare_first_quantization_cost,
    source_projectile_select_first_quantization_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

select = estimate_ftqc_resources(
    source_projectile_select_first_quantization_cost(6, 9, 10, 10, num_bits_nuc_pos=41)
)
assert select.logical.gates.cswap == 774
assert select.logical.gates.toffoli == 2113
assert select.logical.peak_logical_qubits == 382

prepare = estimate_ftqc_resources(
    source_projectile_prepare_first_quantization_cost(6, 8, 10, 10, 10, m_param=524288)
)
adjoint = source_projectile_prepare_first_quantization_adjoint_gate_cost(
    6, 8, 10, 10, 10
)
assert prepare.logical.gates.toffoli == 1128
assert prepare.logical.peak_logical_qubits == 135
assert adjoint.toffoli == 102
```

The projectile PREPARE keeps `plus_t` live through all four calls. Its
adjoint inherits the source unary flag binding failure, so only the
declared adjoint gates are available. Both top-level graphs contain
cost-only leaves. Their reports describe analytical resources.

## SELECT-only block encoding

`SelectBlockEncoding` applies the supplied SELECT on its named
selection and target ports. It keeps the signal preparation and its
normalization available separately. The resource model delegates to the
same SELECT and excludes signal-preparation gates.

```python
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

def select(ports, workspace, classical_bits, adjoint):
    return (Instruction("cx", (*ports["address"], *ports["system"])),)

def prepare(ports, workspace, classical_bits, adjoint):
    return (Instruction("h", ports["address"]),)

signal = BlackBoxPrepare((("address", 1),), (), prepare, normalization=2.0)
encoding = SelectBlockEncoding((("address", 1),), (("system", 1),), select, signal)
circuit = DynamicCircuit(2)
circuit.append(encoding.make_on(selection=(0,), target=(1,)))
report = estimate_ftqc_resources(
    encoding.resource_model(ComponentCost("SELECT", 2, GateCost(clifford=1))),
    emitted_ir=circuit.to_ir(),
)
assert encoding.alpha == 2.0
assert report.emitted is not None
assert report.emitted.unconditional == {"cx": 1}
```

Selection and target ports may be packed or split into named contiguous
words in logical order. Physical wires can be noncontiguous. Work and
classical slots belong to the SELECT implementation supplied by the
caller. The signal state's junk registers are outside the SELECT binding.

## Bounded Pauli SELECT

The explicit bounded binding follows the locked unary traversal. With
three rows, address `11` aliases the third row. The regular `make_on`
binding applies identity to that unused address. Choose the bounded
binding for an address prepared within the declared selection domain.

```python
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import SelectPauliLCU
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

select = SelectPauliLCU(2, 1, ("X", "Z", "X"))
op = select.make_bounded_on(
    selection=(0, 1), target=(2,), workspace=(3,), classical_bits=(0,)
)
circuit = DynamicCircuit(4)
circuit.append(op)
report = estimate_ftqc_resources(op.resource_model(), emitted_ir=circuit.to_ir())
assert report.logical.gates.and_bloq == 1
assert report.logical.peak_logical_qubits == 4
assert report.emitted is not None
assert report.emitted.t_count == 4
assert report.emitted.unconditional["measure"] == 1
```

Work and classical slots follow the declared address width. If the
address is wider than necessary, the source traversal can leave some
of that reserved work idle. The model reports the source peak active
width. The emitted section reports the declared circuit width.

## Qubitization walk for a SELECT block

The walk applies the SELECT block and then reflects about its prepared
signal. The adjoint reverses those calls and adjoints SELECT. Bind the
union of selection ports by name. Shared names require matching widths.
Signal junk and temporary qubits are explicit, and clean workspace can
be reused across stages.

```python
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import SelectPauliLCU
from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

select = SelectPauliLCU(2, 1, ("X", "Z", "X"))

def emit_select(ports, work, bits, adjoint):
    return select.make_bounded_on(
        selection=ports["address"], target=ports["target"],
        workspace=work, classical_bits=bits,
    ).emit()

def prepare(ports, work, bits, adjoint):
    return (Instruction("h", ports["address"][:1]),)

signal = BlackBoxPrepare((("address", 2),), (), prepare, normalization=3.0)
block = SelectBlockEncoding((("address", 2),), (("target", 1),), emit_select, signal)
walk = QubitizationWalk(block)
circuit = DynamicCircuit(5)
circuit.append(walk.make_on(
    selection={"address": (0, 1)}, target=(2,),
    select_workspace=(3,), select_classical_bits=(0,),
    reflection_workspace=(3, 4), reflection_classical_bits=(0,),
))
select_model = select.make_bounded_on(
    selection=(0, 1), target=(2,), workspace=(3,), classical_bits=(0,)
).resource_model()
prepare_model = ComponentCost("Prepare", 2, GateCost(clifford=1))
report = estimate_ftqc_resources(
    walk.resource_model(select_model, prepare_model, prepare_model),
    emitted_ir=circuit.to_ir(),
)
assert report.logical.peak_logical_qubits == 5
assert report.emitted is not None
assert report.emitted.t_count == 8
assert report.emitted.unconditional["measure"] == 2
```

The source places a global scalar of −1 inside the reflection. Native IR
represents it as a phase on the temporary phase target while that wire
is in `|1>`. The source model charges no gate for this scalar. The
emitted report includes the phase instruction.

## Zero-signal LCU block encoding

`LCUBlockEncoding` applies PREPARE, SELECT and PREPARE†. The signal
preparation is identity on the selection registers. Preparation junk
passes through the block. The adjoint uses SELECT† in the middle, and
a controlled block passes the requested polarity to its specialized
SELECT emitter.

```python
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

def select(ports, workspace, classical_bits, adjoint):
    return (Instruction("cx", (*ports["address"], *ports["system"])),)

def prepare(ports, workspace, classical_bits, adjoint):
    return (Instruction("h", ports["address"]),)

preparation = BlackBoxPrepare((("address", 1),), (), prepare, normalization=2.0)
block = SelectBlockEncoding((("address", 1),), (("system", 1),), select, preparation)
lcu = LCUBlockEncoding(block)
circuit = DynamicCircuit(2)
circuit.append(lcu.make_on(selection=(0,), junk=(), target=(1,)))
select_model = ComponentCost("SELECT", 2, GateCost(clifford=1))
prepare_model = ComponentCost("Prepare", 1, GateCost(clifford=1))
report = estimate_ftqc_resources(
    lcu.resource_model(select_model, prepare_model, prepare_model),
    emitted_ir=circuit.to_ir(),
)
assert report.logical.gates.clifford == 3
assert report.logical.peak_logical_qubits == 2
assert report.emitted is not None
assert report.emitted.unconditional == {"h": 2, "cx": 1}
```

Each stage receives its own explicit work and feedback slots. Clean
workspace may be shared across stages. SELECT's named address ports
must be available among the preparation's selection and through-junk
ports with matching widths.

### Walking a zero-signal LCU block

Pass an `LCUBlockEncoding` to the same walk class. Its logical junk is
passed as `block_junk` and stays live through the step. The reflection
uses the identity signal preparation, while the block still runs its
PREPARE and PREPARE†. An LCU preparation work wire can be reused by
the later reflection, including as the phase target.

Continuing the bounded Pauli walk example above:

```python
from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding

lcu_walk = QubitizationWalk(LCUBlockEncoding(block))
circuit = DynamicCircuit(5)
circuit.append(lcu_walk.make_on(
    selection={"address": (0, 1)}, target=(2,),
    select_workspace=(3,), select_classical_bits=(0,),
    reflection_workspace=(3, 4), reflection_classical_bits=(0,),
))
report = estimate_ftqc_resources(
    lcu_walk.resource_model(select_model, prepare_model, prepare_model),
    emitted_ir=circuit.to_ir(),
)
assert report.logical.peak_logical_qubits == 5
assert report.emitted is not None
assert report.emitted.t_count == 8
assert report.emitted.unconditional["measure"] == 2
```

The locked walk signature omits the control port of an already
controlled LCU block. Creating this combination raises an explicit
validation error.

### Applying an external control to the walk

Bind an external control pattern to an uncontrolled block. The walk uses
a specialized positive SELECT emitter and a controlled reflection. A
negative bit conjugates the walk by X. Multiple bits compute one predicate
and erase it after both stages. In the LCU path, PREPARE and PREPARE†
remain unconditional, and the reflection phase acts as Z on the predicate.
Predicate workspace is live through the walk and must be disjoint from
stage workspace.

Continuing the bounded Pauli walk example:

```python
from flagquantum.compiler.registers.control_predicate import ControlRegister

controlled_select = SelectPauliLCU(2, 1, ("X", "Z", "X"), 1)

def emit_controlled(value, ports, work, bits, adjoint):
    assert value == 1
    return controlled_select.make_bounded_on(
        selection=ports["address"], target=ports["target"],
        control=ports["control"], workspace=work, classical_bits=bits,
    ).emit()

control_registers = (ControlRegister(1, (1, 0, 1)),)
circuit = DynamicCircuit(10)
circuit.append(walk.make_on(
    selection={"address": (0, 1)}, target=(2,),
    controls=((3, 4, 5),), control_registers=control_registers,
    controlled_select=emit_controlled,
    control_workspace=(6, 7), control_classical_bits=(2, 3),
    select_workspace=(8, 9), select_classical_bits=(0, 1),
    reflection_workspace=(8,), reflection_classical_bits=(0,),
))
positive_select_model = controlled_select.make_bounded_on(
    selection=(0, 1), target=(2,), control=(7,),
    workspace=(8, 9), classical_bits=(0, 1),
).resource_model()
report = estimate_ftqc_resources(
    walk.resource_model(
        positive_select_model, prepare_model, prepare_model,
        control_registers=control_registers,
    ),
    emitted_ir=circuit.to_ir(),
)
assert report.logical.peak_logical_qubits == 10
assert report.emitted is not None
assert report.emitted.t_count == 20
assert report.emitted.unconditional["measure"] == 5
```

## Walk construction from a Pauli Hamiltonian

`pauli_hamiltonian_walk` in
`flagquantum.algorithms.primitives.pauli_hamiltonian_walk` combines repeated
Pauli products in input order, orders the active target labels, and passes
signed words to bounded SELECT. It converts the requested eigenphase
precision to absolute coefficient precision before building the dense alias
table. Supply `norm_bound` when the source's signed norm estimate is nonreal
or when an independently established bound is needed.

Use the resulting `walk.make_on` with the alias preparation's explicit
selection, signal junk, and workspace requirements. The locked alias
reflection marks the selection register only. Its allocated preparation junk
can remain nonzero, including when it starts at zero. Keep those wires live
after the step. Comparator and reflection work is cleared.

For the two-term example with coefficients +3 and −3 and an explicit norm
bound of √18, the model gives 32 T equivalents and a peak of 14 logical
qubits. Native IR emits 38 T/T† and six measurements, with exact seven-T
Fredkin gates. The allocated circuit has 15 wires including the scalar
phase carrier. Both directions were compared over all measurement histories,
retaining the preparation junk in the source comparison.

### Symbolic alias preparation and periodic Ising walks

`AliasPreparationShape.from_precision` constructs the source register
dimensions and resource formulas before alias values are known. The source
rounds concrete probability precision to at least one bit. Its symbolic
formula retains the unclamped ceiling. Shape-based QROM counts can differ
from counts computed from a filled table.

```python
from sympy import Symbol

from flagquantum.algorithms.primitives.ising_walk import ising_walk
from flagquantum.algorithms.primitives.pauli_hamiltonian_walk import (
    SymbolicPauliHamiltonianWalk,
)

epsilon = Symbol("epsilon", positive=True)
plan, hamiltonian = ising_walk(4, epsilon)
assert isinstance(plan, SymbolicPauliHamiltonianWalk)
assert plan.prepare.normalization == 8
assert plan.gate_cost().cswap == 6
```

The symbolic walk requires a concrete Hamiltonian and estimates gate
categories. The pinned source cannot compute the complete walk's logical
peak when the reflection junk has symbolic width. With a concrete
precision, `ising_walk` returns the bounded Pauli walk and its concrete
alias table, and the full resource model is available.

### Sparse alias costs

`source_sparse_alias_preparation_cost(table)` counts a populated sparse
table. The lookup address covers the support, and the three datasets load
selection, alternate and keep in that order. The full selection width
determines the swap width. For `{3: 1, 0: 3}` over four selections at
precision 0.25, the source reports two CSwap, one And, one measurement,
20 Cliffords and a peak of ten logical qubits. The native circuit emits
18 T/T† instructions using exact seven-T Fredkin gates.

`AliasPreparationShape.from_sparse_precision` estimates a table-free
sparse declaration. Its probability width uses the full selection count
at normalized precision, as specified by the locked source. A populated
table uses the support count at normalized precision. The source composite
has no `sparse_index` port for a single support row, so the source cost query
raises in that case. Native singleton preparation remains executable.

### Sparse chemistry integral data and named lookup

`sparse_chemistry_data` in
`flagquantum.algorithms.primitives.sparse_chemistry_data` prepares the locked
source's ordered integral representatives. One-body lower-triangle entries
are retained, including zeros. The threshold applies only to the two-body
representatives. The full integral tensors determine the normalization.

The 13 datasets contain orbital labels, sign and body flags, their alias
alternates, and keep values. `lookup_ir` loads these through the named
`lookup_output_layout`, including the extra banks for a blocked table.
The caller provides clean output banks and lookup scratch. The banks retain
the selected data and blocked junk. Temporary lookup scratch is cleaned.
`lookup_cost` reports the locked forward QROAM resource model.

Integral preprocessing supports one spatial orbital. The lookup binding
requires nonempty orbital words. The source's full PREPARE fails when it
tries to bind its omitted zero-width target.

### Forward sparse chemistry PREPARE

`PrepareSparseChemistry` binds the uniform address preparation, clean
lookup, probability comparison, signed alias selection and symmetry swaps.
Provide clean selection and output registers in the published layouts.
The lookup banks and comparison flag are retained outputs. The temporary
scratch is reusable after the step.

```python
import numpy as np

from flagquantum.algorithms.primitives.sparse_chemistry_data import sparse_chemistry_data
from flagquantum.algorithms.primitives.sparse_chemistry_prepare import PrepareSparseChemistry
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

one = np.array([[1.0, -0.75], [-0.75, 1.0]])
two = np.zeros((2, 2, 2, 2))
two[0, 0, 0, 0] = -0.5
prepare = PrepareSparseChemistry(
    sparse_chemistry_data(one, two, probability_bits=1, log_block_size=0)
)
layout = (*prepare.selection_layout, *prepare.junk_layout)
width = sum(size for _, size in layout)
ports = partition_register(tuple(range(width)), layout)
work, slots = prepare.requirements()
circuit = DynamicCircuit(width + work)
circuit.append(prepare.make_on(
    ports,
    workspace=tuple(range(width, width + work)),
    classical_bits=tuple(range(slots)),
))
report = estimate_ftqc_resources(prepare.resource_model(), emitted_ir=circuit.to_ir())
assert report.logical.gates.cswap == 8
assert report.emitted is not None
assert report.emitted.t_count == 75
assert report.emitted.unconditional["measure"] == 3
```

At one-bit orbital width, the source's call dictionary assigns the same
key to the flag swap and orbital swaps. Its model counts eight CSwap while
the executable decomposition contains nine Fredkin gates. The emitter and
resource estimate each preserve their corresponding source behavior.

### Sparse chemistry PREPARE reversal

Call `make_coherent_adjoint_on` with the same named ports and scratch
to reverse the preparation on its matching forward output. The inverse
undoes symmetry and alias swaps, phases, comparison, probability Hadamards,
lookup routing and load, then the uniform address preparation. It preserves
coherence on the lookup banks while uncomputing them. Scratch predicates
can still use measurements and immediate feedback.

```python
inverse = prepare.make_coherent_adjoint_on(
    ports,
    workspace=tuple(range(width, width + work)),
    classical_bits=tuple(range(slots)),
)
adjoint_circuit = DynamicCircuit(width + work).append(inverse)
source_cost = estimate_ftqc_resources(prepare.resource_model(adjoint=True))
coherent_cost = estimate_ftqc_resources(
    prepare.resource_model(adjoint=True), emitted_ir=adjoint_circuit.to_ir()
)
assert source_cost.logical == coherent_cost.logical
assert coherent_cost.emitted is not None
assert coherent_cost.emitted.t_count == 75
assert coherent_cost.emitted.unconditional["measure"] == 3
```

The source adjoint model counts its abstract QROAM measurement-erasure
leaf. That leaf has no executable decomposition in the pinned version.
The coherent circuit's counts describe its actual instructions, and
the forward output banks must satisfy their lookup invariant when
uncomputed. The branch checks include a padded lookup with an odd
number of rows.

For the same ports, call `make_measurement_adjoint_on` to consume the
lookup banks by X-basis measurement and apply the resulting phase
corrections before reversing uniform preparation. Supply one extra
classical outcome slot per bank wire.

```python
bank_width = sum(size for _, size in prepare.data.lookup_output_layout)
measured = prepare.make_measurement_adjoint_on(
    ports,
    workspace=tuple(range(width, width + work)),
    classical_bits=tuple(range(slots)),
    bank_bits=tuple(range(slots, slots + bank_width)),
)
measured_circuit = DynamicCircuit(width + work).append(measured)
measured_cost = estimate_ftqc_resources(
    prepare.resource_model(adjoint=True), emitted_ir=measured_circuit.to_ir()
)
assert measured_cost.logical == source_cost.logical
assert measured_cost.emitted is not None
assert measured_cost.emitted.t_count == 75
assert measured_cost.emitted.unconditional["measure"] == 16
```

The executable path retains the coherent path's T count and scratch
width and adds one measurement per lookup bank wire. The separate source
model keeps the upstream abstract erasure formula.

### Bounded sparse chemistry SELECT

Bind the locked SelectSparse stream with named ports and a reusable
accumulator and traversal workspace.

```python
from flagquantum.algorithms.primitives.multiplexers.select_sparse import SelectSparse
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

layout = SelectSparse.source_layout(4)
width = sum(bits for _, bits in layout)
work, slots = SelectSparse.source_requirements(4)
ports = partition_register(tuple(range(width)), layout)
op = SelectSparse.make_source_on(
    4, ports,
    workspace=tuple(range(width, width + work)),
    classical_bits=tuple(range(slots)),
)
circuit = DynamicCircuit(width + work).append(op)
report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
assert report.logical.peak_logical_qubits == 14
assert report.logical.gates.total_t_count() == 40
assert report.emitted is not None
assert report.emitted.t_count == 40
assert report.emitted.unconditional["measure"] == 10
```

The bounded path visits spatial orbitals in ascending order and spin in
least-significant-bit order. It reuses one clean accumulator and two
tree wires across the four Majorana calls for four spin orbitals. For
six spin orbitals it emits 72 T/T† and 18 measurements on 21 wires,
including four clean work wires. The same settings yield 72 T equivalents
and a peak of 21 in the locked logical model.

Use populated orbital addresses with the bounded path. The source tree
can alias padded addresses to populated rows, while the existing
`make_on` selects identity for an unused Majorana address. The optional
source `control_value` adds a pass-through wire for either polarity.
The one-body flag gates the first two Majorana calls. Source resource
queries fail at one spatial orbital, which the bounded binding excludes.

### Sparse chemistry walk with the full preparation signal

`SparseChemistryWalk` composes bounded SELECT with the reflection
$2|L\rangle\langle L|-I$, where `L` is the state prepared across all
PREPARE registers. The caller retains those registers between steps.
SELECT uses the orbital and body-flag wires among them. The reflection
uses coherent PREPARE reversal, a zero mark over the entire signal, and
forward PREPARE. Its phase target and unary workspace return to zero.
The adjoint reverses SELECT's Majorana order and applies the reflection
first.

Using the unblocked two-spatial-orbital `prepare` above:

```python
from flagquantum.algorithms.primitives.sparse_chemistry_walk import SparseChemistryWalk

walk = SparseChemistryWalk(prepare)
walk_width = sum(size for _, size in walk.layout)
walk_ports = partition_register(tuple(range(walk_width)), walk.layout)
walk_work, walk_slots = walk.requirements()
step = walk.make_on(
    walk_ports,
    workspace=tuple(range(walk_width, walk_width + walk_work)),
    classical_bits=tuple(range(walk_slots)),
)
walk_circuit = DynamicCircuit(walk_width + walk_work).append(step)
walk_report = estimate_ftqc_resources(
    walk.source_resource_model(), emitted_ir=walk_circuit.to_ir()
)
assert walk.alpha == 3.75
assert walk_report.logical.gates.total_t_count() == 152
assert walk_report.logical.gates.measurement == 22
assert walk_report.logical.peak_logical_qubits == 34
assert walk_report.emitted is not None
assert walk_report.emitted.t_count == 278
assert walk_report.emitted.unconditional["measure"] == 38
assert walk_report.emitted.declared_qubits == 50
```

The source model counts a narrower reflection and an abstract QROAM
erase. Its adjoint consumes the bank ports before the zero mark, while
the comparison flag stays live. The original erase leaf has no gate
decomposition. Using the verified coherent inverse and preserving the
source mark and fresh-junk layout leaves a nonzero probability of
occupied junk on zero input: 3/64 for the unblocked example and 1/32
with two rows per block. The executable full-signal reflection keeps
those registers logical and marks their zero state as well.

### THC uniform address preparation

`THCUniformPreparation` prepares the address domain used by the alias
table. It assigns equal positive amplitudes to the upper triangle
`mu <= nu < num_mu` and the one-body column `nu == num_mu` with
`mu < num_spin_orb // 2`. The one-body flag retains the source port
name `nu_eq_mp1`. On zero input the success flag is one and the rotation
wire and temporary workspace return to zero. The circuit uses one or two
exact amplitude-amplification steps according to the domain fraction.

```python
from flagquantum.algorithms.primitives.thc_uniform_preparation import THCUniformPreparation
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

prep = THCUniformPreparation(2, 4)
width = sum(size for _, size in prep.layout)
ports = partition_register(tuple(range(width)), prep.layout)
work, slots = prep.requirements()
op = prep.make_on(
    ports,
    workspace=tuple(range(width, width + work)),
    classical_bits=tuple(range(slots)),
)
circuit = DynamicCircuit(width + work).append(op)
report = estimate_ftqc_resources(prep.source_resource_model(), emitted_ir=circuit.to_ir())
assert report.logical.gates.and_bloq == 40
assert report.logical.gates.toffoli == 4
assert report.logical.peak_logical_qubits == 16
assert report.emitted is not None
assert report.emitted.t_count == 152
assert report.emitted.unconditional["ccx"] == 8
assert report.emitted.unconditional["ry"] == 3
assert report.emitted.unconditional["measure"] == 38
assert report.emitted.declared_qubits == 17
```

The locked source graph flags `nu == num_mu + 1` and requires that
condition together with `nu < num_mu` to set success from zero. Its
greater-than leaf supplies costs only. Its last X also flips the junk
bit immediately before a clean Free. The source model follows those
original calls. The executable circuit uses the alias table's sentinel
and a clean predicate for the union of the two domains. The report
therefore keeps source logical costs separate from emitted instructions.

### THC alias rows and named lookup

`thc_chemistry_data` constructs the five source columns in row-major
upper-triangle order and appends the one-body eigenvalues. It uses that
ordering for alias sampling and preserves the normalized-zeta and
eigenvalue contributions to the source normalization.

`lookup_table()` permutes the resulting rows into the address order
`nu * (nu + 1) // 2 + mu`. Alternate coordinates, sign bits and keep
values move together. Use `lookup_table(source_order=True)` or
`lookup_cost(source_order=True)` for the original row declaration.
The unblocked layout has only the five selected words. The blocked
layout also binds retained words in dataset and cell order.

```python
import numpy as np

from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.core.ir import CircuitIR
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

matrix = np.arange(1, 17).reshape(4, 4)
data = thc_chemistry_data(
    [2.0, -3.0], np.ones((4, 2)), (matrix + matrix.T) / 2,
    probability_bits=4, log_block_size=1,
)
assert data.normalization == 277
address = tuple(range(data.address_width))
layout = data.lookup_output_layout
width = len(address) + sum(size for _, size in layout)
ports = partition_register(tuple(range(len(address), width)), layout)
work = data.lookup_table().workspace_size(0, (len(address),))
gates = data.lookup_ir(address, ports, tuple(range(width, width + work)), tuple(range(work)))
report = estimate_ftqc_resources(
    data.lookup_cost(), emitted_ir=CircuitIR(width + work, gates).validate()
)
assert report.logical.gates.total_t_count() == 64
assert report.emitted is not None
assert report.emitted.t_count == 64
assert report.emitted.unconditional["measure"] == 4
assert report.emitted.declared_qubits == 30
```

Call `lookup_ir(..., uncompute=True)` with matching signed bank outputs
to reverse the routing and clear the loaded words. The tests compare
forward and reverse gate dependencies with the pinned QROAM using the
same permuted inputs, including every measurement history of the small
blocked and unblocked fixtures. The original preprocessing uses the
unpermuted declaration and has its own model comparison.

### Full THC alias preparation

`PrepareTHCChemistry` binds the uniform signal, contiguous index,
named lookup and signed alias exchange. The one-body flag is first
cleared from the original pair, then written from the selected pair.
That flag controls the final source-style symmetrization together with
`plus_a`. The original address, lookup banks and symmetry predicate
remain logical outputs. The comparison flag and temporary work return
to zero. Bind the inverse with `adjoint=True` and the same ports.

```python
from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

data = thc_chemistry_data(
    [6.0, -8.0], [[1.0, 1.0], [1.0, 1.0]], [[1.0, -2.0], [-2.0, 4.0]],
    probability_bits=1, log_block_size=1,
)
prepare = PrepareTHCChemistry(data)
layout = (*prepare.selection_layout, *prepare.junk_layout)
width = sum(size for _, size in layout)
ports = partition_register(tuple(range(width)), layout)
work, slots = prepare.requirements()
op = prepare.make_on(
    ports, workspace=tuple(range(width, width + work)), classical_bits=tuple(range(slots))
)
circuit = DynamicCircuit(width + work).append(op)
report = estimate_ftqc_resources(prepare.source_resource_model(), emitted_ir=circuit.to_ir())
assert report.logical.gates.total_t_count() == 286
assert report.logical.peak_logical_qubits == 32
assert report.emitted is not None
assert report.emitted.t_count == 266
assert report.emitted.unconditional["ccx"] == 8
assert report.emitted.unconditional["ry"] == 3
assert report.emitted.unconditional["measure"] == 49
assert report.emitted.declared_qubits == 40
```

The locked model retains the source uniform preparation, row order and
one-body flag wiring. Its contiguous index leaf supplies a Toffoli
formula and port width, while the executable index uses the shared
weighted-column XOR circuit. The pinned unblocked PREPARE has gate
costs but fails to bind the omitted `junk_target0_` port in its
decomposition. Use `source_gate_cost()` for those counts. The native
unblocked preparation and coherent inverse support the five selected
words without extra banks.

### THC walk with the full preparation signal

The executable `THCChemistryWalk` binds the supplied Givens networks
to SELECT and reflects about the full signed PREPARE state. It keeps
the prepared ports live between steps and cleans the temporary work.
Its inverse applies the same involutory SELECT after the reflection.
Use exact angles with no SELECT workspace, or pass the existing
eight-bit lookup workspace size. For the measured path SELECT owns
feedback slots above the existing circuit and reflection slots.

Using the blocked two-index `prepare` above:

```python
from math import tau

from flagquantum.algorithms.primitives.thc_chemistry_walk import THCChemistryWalk

walk = THCChemistryWalk(
    prepare,
    networks=(((0, 1, tau / 8),), ((0, 1, -tau / 16),)),
    one_body_networks=((), ((0, 1, tau / 32),)),
    select_workspace_qubits=26,
    measurement_erasure=True,
)
width = sum(size for _, size in walk.layout)
work, slots = walk.requirements()
ports = partition_register(tuple(range(width)), walk.layout)
step = walk.make_on(
    ports, workspace=tuple(range(width, width + work)), classical_bits=tuple(range(slots))
)
circuit = DynamicCircuit(width + work).append(step)
report = estimate_ftqc_resources(walk.source_resource_model(), emitted_ir=circuit.to_ir())
assert report.logical.gates.total_t_count() == 1056
assert report.logical.gates.measurement == 97
assert report.logical.peak_logical_qubits == 36
assert report.emitted is not None
assert report.emitted.t_count == 760
assert report.emitted.unconditional["measure"] == 187
assert report.emitted.declared_qubits == 64
```

The model uses the locked narrower reflection and the directional PREPARE
costs. In the source adjoint, the QROAM bank is consumed by an opaque
measurement-erase leaf. The executable walk uses coherent PREPARE
reversal and a zero mark on all of its signal wires. Its gate stream
therefore includes the SELECT implementation, the wider mark and the
coherent lookup inverse. Gate-only source queries remain available when
the unblocked source peak query fails.

## Ordered Trotter evolution

`TrotterProduct` accepts a term list and an explicit product schedule. Each
term owns its angle-dependent gate stream and resource declaration on the
same named through ports. The product binds each coefficient as
`2 * coefficient * timestep`, visits paired entries in order, and stops
at the shorter sequence. The binding accepts named circuit parameters.
All physical wires are caller-owned and ordered most significant first.

```python
from flagquantum.algorithms.primitives.ising_evolution import (
    IsingXUnitary,
    IsingZZUnitary,
)
from flagquantum.algorithms.primitives.trotter_product import TrotterProduct
from flagquantum.circuit import Circuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

product = TrotterProduct(
    (IsingXUnitary(3), IsingZZUnitary(3)),
    (0, 1, 0),
    (0.05, 2, 0.05),
    0.125,
)
planned = estimate_ftqc_resources(product)
op = product.make_on(system=(0, 1, 2))
circuit = Circuit(3).append(op)
report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
assert planned.logical == report.logical
assert report.logical.gates.clifford == 6
assert report.logical.gates.rotation == 9
assert report.logical.peak_logical_qubits == 3
assert report.emitted is not None
assert dict(report.emitted.unconditional) == {"rx": 6, "cx": 6, "rz": 3}
```

The inverse reverses and adjoints the full emitted stream. The modeled Ising
gate categories and peak width apply in either direction. The periodic ZZ
term retains both directed edges on a two-site ring. A one-site ZZ ring is
rejected because the source edge gives CNOT the same control and target.
These terms declare no work wires or measurements. The source ZZ rotation
tolerance is synthesis metadata, while the native IR keeps an exact Rz.

## Square-lattice Hubbard resource schedules

The Hubbard interaction and hopping terms in Qualtran v0.7.0 declare cost
graphs. The native model preserves both plaquette colors, the full and
stripped second-order schedules, and the Hamming-weight phasing alternative.
It keeps the original `2 * length` signature width as the starting point
for the source maximum-of-callees fallback. A physical square lattice with
two spin sectors uses `2 * length**2` system wires.

```python
from flagquantum.algorithms.primitives.hubbard_trotter import source_hubbard_suzuki_cost
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

model = source_hubbard_suzuki_cost(4, 4.0, 0.1234, hamming_weight=True)
report = estimate_ftqc_resources(model)
assert report.logical.gates.t == 192
assert report.logical.gates.and_bloq == 70
assert report.logical.gates.clifford == 2038
assert report.logical.gates.rotation == 40
assert report.logical.gates.measurement == 70
assert report.logical.peak_logical_qubits == 19
assert report.logical.gates.total_t_count() == 912
assert report.emitted is None
```

The Hamming-weight phase graph charges repeated copies of the same Z power.
The executable phasing primitive uses a different power for each bit of the
computed weight. The one-bit source width query fails on an empty junk
register. The native model derives its width from the two bound qubits.

### Executable hopping plaquette

`HubbardPlaquette` implements the four-mode hopping of Appendix E in
[Early fault-tolerant simulations of the Hubbard model](https://arxiv.org/abs/2012.09238).
Supply four consecutive Jordan–Wigner modes in cyclic site order. The wire
labels can be remapped by the caller. Two Fourier pairs enter the rotated
basis, where a Clifford maps the XX and YY terms to two equal-angle Rz
rotations. The inverse basis restores the original mode order.

```python
from flagquantum.algorithms.primitives.hubbard_plaquette import HubbardPlaquette
from flagquantum.circuit import Circuit

op = HubbardPlaquette.make_on(modes=(0, 1, 2, 3), timestep=0.123)
circuit = Circuit(4).append(op)
report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
assert report.logical.gates.t == 8
assert report.logical.gates.rotation == 2
assert report.emitted is not None
assert report.emitted.t_count == 8
assert report.emitted.unconditional["rz"] == 6
```

The extra four emitted Rz instructions have zero angle and belong to the
reused Fourier decomposition. Emission includes the fermionic routing and
the central Clifford basis change. The source plaquette resource graph
counts the Fourier and phase calls separately. The implementation uses no
work wires or measurements. Complete complex matrices are tested against an
independent Jordan–Wigner Hamiltonian.

### Executable periodic Suzuki step

The lattice binding uses the paper's shifted on-site interaction and
periodic plaquettes. Supply two row-major spin registers, each with
`length**2` wires. At length two both colors contribute the same square,
counting periodic bonds with multiplicity. Larger even lattices partition
the nearest-neighbor edges between the colors.

```python
from flagquantum.algorithms.primitives.hubbard_lattice import HubbardSuzukiStep
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

op = HubbardSuzukiStep.make_on(
    length=2,
    up=(0, 1, 2, 3),
    down=(4, 5, 6, 7),
    hubb_u=2.1,
    timestep=0.123,
    hamming_weight=True,
    workspace=(8, 9, 10),
)
circuit = DynamicCircuit(11).append(op)
report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
assert report.emitted is not None
assert report.emitted.t_count == 88
assert report.emitted.unconditional["measure"] == 10
assert report.emitted.declared_qubits == 11
assert report.logical.peak_logical_qubits == 5
```

Workspace starts and ends at zero for each phase batch. A feedback slot is
reused after its correction and allocated above any preexisting classical
bits. The Hamming-weight phases include the scalar correction needed to
recover the absolute Rz phase. The native lattice uses the paper's physical
angles, while the separately reported model preserves the locked source's
angle and width conventions. Set `adjoint=True` to reverse the layers and
negate their physical angles, including in the measured construction.

## Grid kinetic phase

The grid kinetic model preserves the locked per-electron order of
SumOfSquares, QuantumVariableRotation and Free. The source phase leaf
declares a generic rotation per result bit. The native phase binding uses
a caller-specified angle in radians per unit of the unsigned square sum.
It applies the square XOR again after phasing to clear the result.

```python
from flagquantum.algorithms.primitives.grid_kinetic import GridKineticEnergy
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

op = GridKineticEnergy.make_on(
    grid_points=1,
    system=(((0,), (1,), (2,)), ((3,), (4,), (5,))),
    result=(6, 7, 8, 9),
    workspace=(10,),
    radians_per_unit=0.317,
)
circuit = DynamicCircuit(11).append(op)
report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
assert report.logical.gates.toffoli == 2
assert report.logical.peak_logical_qubits == 10
assert report.emitted is not None
assert report.emitted.t_count == 16
assert report.emitted.unconditional["measure"] == 4
```

The three coordinate words per electron are big endian and follow the
unsigned input type of the source square sum. The source sizes each word
with an extra bit. The native circuit preserves that bit as part of the
unsigned value. Result and workspace start at zero, are reused between
electrons, and return to zero. The measurement slot is chosen above any
existing feedback bits. The inverse reverses the electron order and
negates the phase coefficient.

## Grid pair potential

The locked model preserves the source add and erase calls, squared-distance
register width, four-bank coefficient lookup, cubic and Newton estimates,
and one call for each upper-triangular electron pair. The original coefficient
table occupies a prefix of the wider distance address. The peak uses the
source scheduler's early release of the system and coefficient wires.

The executable potential binds unsigned, big-endian axis words and computes
the magnitude of each signed difference before squaring. It phases the shared
C1/Newton fixed-point inverse-distance estimate and erases both the estimate
and the squared distance. Coincident points yield zero phase. The locked
cubic/Newton graph and the executable C1/Newton circuit use different seed
arithmetic and have separate resource reports.

```python
from flagquantum.algorithms.primitives.grid_potential import (
    GridPotentialEnergy,
    grid_potential_requirements,
)
from flagquantum.algorithms.primitives.grid_potential_resource_model import (
    source_grid_potential_cost,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

work, slots = grid_potential_requirements(1, precision=4, iterations=1)
op = GridPotentialEnergy.make_on(
    grid_points=1,
    system=(((0,), (1,), (2,)), ((3,), (4,), (5,))),
    workspace=tuple(range(6, 6 + work)),
    radians_per_unit=0.721,
    precision=4,
)
circuit = DynamicCircuit(6 + work).append(op)
report = estimate_ftqc_resources(
    source_grid_potential_cost(2, 1), emitted_ir=circuit.to_ir()
)
assert (work, slots) == (177, 13)
assert report.logical.peak_logical_qubits == 81
assert report.emitted is not None
assert report.emitted.t_count == 3104
assert report.emitted.unconditional["measure"] == 776
```

### Cubic inverse distance

Set `cubic=True` in both `grid_potential_requirements` and
`GridPotentialEnergy.make_on` to select the piecewise cubic followed by
one corrected Newton step from [Eqs. 36–49](https://arxiv.org/pdf/2308.12352).
The executable QROM loads four coefficient magnitudes and the interval's
Newton correction. Leading-one selection supplies the normalized interval
offset. Three reversible Horner stages compute each complete product before
truncating it. Phasing is followed by the inverse arithmetic and QROM erase.

The locked coefficient table and resource model retain their original
address conventions. The executable table covers the complete distance
register and uses the paper's intervals, including separate rows for
distances two and three. Coefficients and the Newton correction are
quantized from the published decimal constants.

For the two-electron example above, use five fractional bits and
`cubic=True`. This requires 212 work qubits and emits 4,136 T/T† instructions
and 1,034 measurements. The locked source model still reports a peak of 81
logical qubits. These figures include reversible clearing in the emitted
circuit.

### Grid notebook resource queries

Run `python -m examples.algorithms.grid_hamiltonian_resources` to reproduce
the locked grid notebook's component resource queries without a notebook
runtime. The example keeps QEC categories and the historical T-complexity
convention separate. The historical calculation first enables the legacy
Clifford shims for And compute and erase in addition and QROM, then folds
the resulting categories into T, Clifford, and rotation counts.

For the notebook's two electrons and 21 points per axis, the pair potential
has 9,176 historical T gates before rotation synthesis, 24 rotations and
2,646 historical Cliffords. Counting each rotation at the historical
`1e-11` synthesis tolerance gives 10,424 T gates. The default QEC conversion
uses 11 T equivalents per rotation and gives 9,440 T equivalents. These are
model values. The cubic execution example reports separately counted IR.

## Single-factorization PREPARE and SELECT models

The chemistry QROAM placeholders retain the original analytical costs.
The two-axis model counts blocks along both axes and reports the original
signature width, which includes only the first address. The default
one-axis forward optimizer requires at least as many rows as target bits.

```python
from flagquantum.algorithms.primitives.single_factorization_resource_model import (
    source_single_factorization_inner_prepare_cost,
    source_single_factorization_outer_prepare_cost,
    source_single_factorization_select_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

inner = estimate_ftqc_resources(
    source_single_factorization_inner_prepare_cost(50, 10, 12, 7, 4, 8)
)
outer = estimate_ftqc_resources(source_single_factorization_outer_prepare_cost(50, 12))
select = estimate_ftqc_resources(source_single_factorization_select_cost(10))
assert inner.logical.gates.toffoli == 682
assert inner.logical.peak_logical_qubits == 60
assert outer.logical.gates.toffoli == 46
assert outer.logical.gates.rotation == 2
assert select.logical.gates.toffoli == 16
assert select.logical.peak_logical_qubits == 9
```

These source components provide resource graphs. SELECT's
`additional_control` parameter leaves the locked signature and estimate
unchanged. The model does not imply that the control acts on a physical
system register.

### Single-factorization block encoding

The resource graph includes the directional PREPARE costs. In the
one-body block, the inverse can use a different two-axis blocking pair.
The full block calls outer PREPARE, the one-body block, a controlled
zero reflection, the one-body block again, and outer PREPARE adjoint.
The reflection's only control in the locked decomposition is `succ_l`.

```python
from flagquantum.algorithms.primitives.single_factorization_encoding_model import (
    source_single_factorization_block_encoding_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

report = estimate_ftqc_resources(
    source_single_factorization_block_encoding_cost(50, 10, 12, 7)
)
assert report.logical.gates.toffoli == 3095
assert report.logical.gates.cswap == 50
assert report.logical.gates.measurement == 179
assert report.logical.peak_logical_qubits == 80
```

The locked PREPARE and SELECT leaves supply resource graphs. The source
leaves `alpha` and `epsilon` undefined. The top-level resource report
describes that model and its original binding conventions.

Run the 8-bit single-factorization estimate using dimensions alone:

```bash
python -m examples.algorithms.ftqc_resource_estimation --single-factorization 10 50 12
```

The default blocking gives 3,103 Toffolis, 50 controlled swaps, 179
measurements, 13,372 T equivalents, and a peak of 80 logical qubits.
With total error budget `1e-3` and physical error `1e-4`, the Beverland fast
data block uses 30,294 physical qubits.

### Executable single-factorization SELECT

`SelectSingleFactorization` implements Eq. A4 using the selected Majorana
sequence in Fig. 13 of [arXiv:2011.03494](https://arxiv.org/abs/2011.03494).
It uses orbital-major, spin-minor system order. Spatial addresses must be
populated. Both success flags activate the selected operator. A caller can
bind the nonzero-factor flag for the second SELECT in Fig. 15. The caller
provides clean workspace, and the dynamic circuit assigns feedback slots.

```python
from flagquantum.algorithms.primitives.multiplexers.select_single_factorization import (
    SelectSingleFactorization,
)
from flagquantum.dynamic import DynamicCircuit

op = SelectSingleFactorization(
    p=(0,), q=(1,), spin=2, succ_pq=3, succ_l=4,
    system=(5, 6, 7, 8), workspace=(9, 10, 11, 12),
)
circuit = DynamicCircuit(13)
circuit.append(op)
gates = circuit.to_ir().instructions
assert sum(g.name in {"t", "tdg"} for g in gates) == 28
assert sum(g.name == "measure" for g in gates) == 7
```

Four spin orbitals with the additional control emit 32 T-family instructions
and eight measurements with five work wires. The locked source declares four
Toffolis for either control configuration and leaves SELECT undecomposed.
The emitted counts include activation and measurement-based uncomputation.

### Executable outer single-factorization PREPARE

The B8 marginal uses twice the absolute triangular one-body weight and the
square of each factor's absolute triangular weight. The executable operation
prepares the uniform factor register, loads the alias and nonzero-factor words,
and swaps both together. Exact uniform preparation sets the success flag
deterministically. The alias flag and table words are retained until the
adjoint. Both directions return the borrowed scratch to zero.

```python
from flagquantum.algorithms.primitives.single_factorization_outer_prepare import (
    PrepareSingleFactorizationOuter,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

prepare = PrepareSingleFactorizationOuter.from_coefficients(
    (0.0, -0.5, 0.0),
    ((0.0, -1.0, 0.0), (0.0, -2.0, 0.0), (0.0, -1.0, 0.0)),
    probability_bits=1,
    log_block_size=1,
)
logical = sum(size for _, size in prepare.layout)
work, _ = prepare.requirements()
ports = partition_register(tuple(range(logical)), prepare.layout)
scratch = tuple(range(logical, logical + work))
circuit = DynamicCircuit(logical + work)
circuit.append(prepare.make_on(ports, workspace=scratch))
circuit.append(prepare.make_on(ports, workspace=scratch, adjoint=True))
```

The four-row fixture emits 45 T-family instructions and one measurement per
direction, with two clean work wires and 15 live logical wires. The locked
outer PREPARE supplies only its resource graph and its automatic chemistry
blocking model rejects these small tables. The executable adjoint reverses the
selected forward block shape, including its relative phases.

### Executable inner single-factorization PREPARE

The B11 preparation keeps the outer factor index read-only. It prepares the
exact uniform triangle, computes the zero-based address `q*(q+1)//2+p`,
and looks up the alias data on both axes. The swap exchanges both coordinates
and their sign. Apply a phase using the selected `theta` between PREPARE and
its adjoint to incorporate the Hamiltonian coefficient sign. The rotation
wire and borrowed scratch end at zero. The address, alias words, and route
banks are retained until unpreparing.

Bind `PrepareSingleFactorizationInner.from_coefficients` with the named
`layout` and `requirements()`, using the same `make_on` and
`circuit.append(op)` convention as the outer preparation. Coefficients in
each factor are ordered by the same zero-based triangular address. In the
three-spatial-orbital, three-factor-row fixture with one probability bit and
two cells per axis, each direction emits 219 T-family instructions and 25
measurements and borrows seven clean wires. The locked source provides a
resource graph for the top-level operation. The emitted stream uses exact
uniform amplification and the matching inverse of the selected QROAM banks.

### Executable single-factorization block encoding

The executable sequence follows Fig. 15. The outer alias prepares the factor,
and the inner alias prepares the signed triangular address. The first SELECT
runs on every successful factor. The middle reflection and the second SELECT
run only on nonzero factors. The SELECT phase uses the same activation
predicate as its Majorana pair.

If $A_j$ is the alias-quantized signed spin and order average for factor
$j$, and $w_j$ is its quantized outer probability, the zero-signal block
is $w_0 A_0 + \sum_{j>0} w_j(2 A_j^2-I)$. The identity term records the
constant shift from the reflection.

```python
from flagquantum.algorithms.primitives.single_factorization_block_encoding import (
    SingleFactorizationBlockEncoding,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

encoding = SingleFactorizationBlockEncoding.from_coefficients(
    2, (1.0, -2.0, 1.0), ((2.0, 3.0, -1.0),), outer_bits=1, inner_bits=1,
)
logical = sum(width for _, width in encoding.layout)
work, _ = encoding.requirements()
ports = partition_register(tuple(range(logical)), encoding.layout)
circuit = DynamicCircuit(logical + work)
circuit.append(encoding.make_on(ports, workspace=tuple(range(logical, logical + work))))
```

This fixture emits 408 T-family instructions and 67 measurements with six
clean scratch qubits. The locked resource model above retains its original
signature and reflection control. The paper-backed executable circuit also
reflects on the inner alias sample and uses the nonzero-factor control.

## Double-factorization PREPARE and programmed rotation

The declared models reproduce the locked inner PREPARE, outer PREPARE,
indexed data and programmed rotation graphs. They preserve the original
port widths and the direction of the chemistry QROAM child. For 50 factors,
10 spin orbitals, 250 retained eigenvalues, seven amplitude-amplification
bits, 12 alias bits and eight basis-rotation bits, the modeled Toffoli counts
are 159/76 for inner PREPARE, 62/33 for outer PREPARE, 45/15 for indexed data
and 235/99 for programmed rotation, in forward/adjoint order. The inner
PREPARE also models seven Ands and seven measurements for its offset Add.

The executable indexed-data lookup follows Eq. C29 and returns the
nonzero-factor flag, rank minus one, the supplied rotation word and the
offset into the concatenated eigenvalue rows. The rank-minus-one convention
uses the available rank width even at full rank. Block banks are retained
until the matching inverse.

```python
from flagquantum.algorithms.primitives.double_factorization_indexed_data import (
    DoubleFactorizationIndexedData,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

data = DoubleFactorizationIndexedData(
    ranks=(4, 2, 1, 4, 3), rotation_words=(1, 3, 5, 7, 0),
    rotation_bits=3, log_block_size=1,
)
logical = sum(width for _, width in data.layout)
ports = partition_register(tuple(range(logical)), data.layout)
scratch = tuple(range(logical, logical + data.workspace_size()))
circuit = DynamicCircuit(logical + len(scratch))
circuit.append(data.make_on(ports, workspace=scratch))
circuit.append(data.make_on(ports, workspace=scratch, adjoint=True))
```

The indexed-data stream uses the concrete locked QROAMClean decomposition.
The chemistry QROAM cost placeholder remains a separate model.

### Ordered double-factorization block encodings

The ordered model follows the locked block's outer PREPARE, indexed-data
load, one-body block, reflection, repeated one-body block, and inverse
loads. The middle reflection touches the orbital selection and spin, and
the locked circuit controls it on the outer success flag. Its one-body
number phase applies to the first orbital of the selected spin register.
The custom source gate-count graph uses a fixed five-orbital word for this
phase. The ordered model uses the decomposition's actual spatial word
width for lifetimes. Both widths give the same declared gate cost.

```python
from flagquantum.algorithms.primitives.double_factorization_encoding_model import (
    source_double_factorization_block_encoding_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

model = source_double_factorization_block_encoding_cost(
    auxiliaries=50, spin_orbitals=10, eigenvalues=250,
    alias_bits=12, basis_rotation_bits=8,
)
resources = estimate_component_resources(model)
assert resources.gates.toffoli == 1304
assert resources.peak_logical_qubits == 101
```

The shared emitted number-phase predicate uses four T-family instructions,
one measurement and one clean work wire with two success controls. A third
control for the nonzero-factor flag uses eight T-family instructions, two
measurements and two work wires. These are leaf-level emitted counts. The
full encoding above is an analytical model.

### Executable outer double-factorization PREPARE

Outer PREPARE samples a factor with coherent aliases, preserving the sample,
comparison bit and loaded banks for the adjoint. Supply the factor weights
directly, or build them from the eigenvalues of the effective one-body matrix
and retained factors. The latter uses the one-body trace norm and one fourth
of the squared absolute eigenvalue sum of each remaining factor, as in
Eq. C11. Factor zero is the one-body term.

```python
from flagquantum.algorithms.primitives.double_factorization_outer_prepare import (
    PrepareDoubleFactorizationOuter,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

prepare = PrepareDoubleFactorizationOuter.from_weights(
    (1, 3, 1, 7), probability_bits=1, log_block_size=1,
)
logical = sum(size for _, size in prepare.layout)
work, _ = prepare.requirements()
ports = partition_register(tuple(range(logical)), prepare.layout)
scratch = tuple(range(logical, logical + work))
circuit = DynamicCircuit(logical + work)
circuit.append(prepare.make_on(ports, workspace=scratch))
circuit.append(prepare.make_on(ports, workspace=scratch, adjoint=True))
```

Each direction in this fixture emits 30 T-family instructions and one
measurement and uses two scratch qubits. The lookup stream follows the
concrete locked QROAMClean. The top-level Qualtran outer PREPARE is a
resource declaration.

### Rank-conditioned uniform preparation

The shared rank preparation follows Appendix C step 3(a) of
[arXiv:2011.03494](https://arxiv.org/abs/2011.03494). A reversible prefix
computes which Hadamards are active from the leading bits of rank minus one.
The read-only angle word specifies a binary fraction of one full turn.
The success output marks an in-range index with the rotation qubit zero,
so residual failure from the finite rotation precision remains coherent.

With a two-bit rank and a four-bit angle, each direction emits 64 T-family
instructions, 16 measurements and 12 controlled rotations. It uses seven
clean scratch qubits. The controlled Hadamards and rotations are native
IR operations. Their synthesis cost belongs to a separate resource model.

### Executable inner double-factorization PREPARE

The inner alias binds the rank, offset and rotation words returned by the
indexed data loader. It computes the contiguous address by adding the
prepared local orbital to the factor offset. A coherent alias comparison
can exchange the orbital and its sign on a successful preparation branch.
The failure branch preserves its orbital. The address and lookup banks
remain live until the matching adjoint.

```python
from flagquantum.algorithms.primitives.double_factorization_inner_prepare import (
    PrepareDoubleFactorizationInner,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

prepare = PrepareDoubleFactorizationInner.from_eigenvalues(
    (1.0, -3.0), ((-2.0,),), probability_bits=1, rotation_bits=2,
)
metadata = prepare.indexed_data()
layout = (*metadata.layout, *(port for port in prepare.layout if port[0] not in dict(metadata.layout)))
logical = sum(width for _, width in layout)
ports = partition_register(tuple(range(logical)), layout)
work, _ = prepare.requirements()
scratch = tuple(range(logical, logical + work))
circuit = DynamicCircuit(logical + work)
circuit.append(metadata.make_on(
    {name: ports[name] for name, _ in metadata.layout},
    workspace=scratch[:metadata.workspace_size()],
))
circuit.append(prepare.make_on(
    {name: ports[name] for name, _ in prepare.layout}, workspace=scratch,
))
```

The two-factor fixture emits 54 T-family instructions, ten measurements and
six native controlled rotations per inner direction. It uses three clean
scratch wires. The chemistry resource placeholder is counted separately.

### Executable double-factorization number SELECT

The SELECT computes a fresh contiguous index from the factor offset and the
post-alias orbital. The loaded binary angle words program adjacent rotations
on a spin-major system. Each pair uses the commuting XY and YX rotations
from Eq. 62 of [arXiv:2007.14460](https://arxiv.org/abs/2007.14460), lowered
through a shared phase gradient and measured addition. The number phase and
coefficient sign use the same success predicate. Reversing the basis, spin
swap, lookup and address computation clears the entire clean bank.

```python
from flagquantum.algorithms.primitives.multiplexers.select_double_factorization import (
    SelectDoubleFactorization,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

select = SelectDoubleFactorization.from_angle_words(
    2, ((1,), (2,), (3,), (4,)), angle_bits=4,
)
logical = sum(width for _, width in select.layout())
work = select.workspace_size()
ports = partition_register(tuple(range(logical)), select.layout())
circuit = DynamicCircuit(logical + work)
circuit.append(select.make_on(ports, workspace=tuple(range(logical, logical + work))))
```

The two-orbital fixture emits 104 T-family instructions and 19 measurements
with three scratch wires. The three-orbital blocked fixture emits 218
T-family instructions and 28 measurements with three scratch wires. The
locked DF programmed-rotation component remains a separate cost declaration.

### Executable double-factorization block encoding

The executable sequence follows Fig. 16 of
[arXiv:2011.03494](https://arxiv.org/abs/2011.03494). It prepares the factor
alias, loads the indexed metadata, and applies two signed inner
PREPARE–SELECT–adjoint bodies around the middle reflection. Both the
middle reflection and the second SELECT require a nonzero factor and
successful outer preparation. The reflection covers the orbital, alias
sample, rotation and spin signal wires.

The effective one-body input uses the negative Z coefficient from
the number operator. The zero-signal block includes the constant shift
from the middle reflection. At finite rank-rotation precision a failed
inner preparation contributes the identity on that branch. For an outer
factor probability $w_j$ and projected inner body $A_j$, the circuit encodes
$w_0 A_0 + \sum_{j>0} w_j(2A_j^2-I)$.

```python
from flagquantum.algorithms.primitives.double_factorization_block_encoding import (
    DoubleFactorizationBlockEncoding,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

encoding = DoubleFactorizationBlockEncoding.from_eigenvalues(
    (1.0, -3.0), ((2.0,),), ((1,), (3,), (2,)),
    outer_bits=1, inner_bits=1, rank_rotation_bits=2, basis_rotation_bits=4,
)
logical = sum(width for _, width in encoding.layout)
work, _ = encoding.requirements()
ports = partition_register(tuple(range(logical)), encoding.layout)
circuit = DynamicCircuit(logical + work)
circuit.append(encoding.make_on(ports, workspace=tuple(range(logical, logical + work))))
```

Angle rows are provided as fixed-point words in concatenated eigenvalue
order. To derive the words from real eigenvectors, pass one sequence of
eigenvalue and unit eigenvector pairs for the one-body term and one sequence
per retained factor. Each vector spans the same spatial orbital basis.

```python
from math import sqrt

diagonal = 1 / sqrt(2)
encoding = DoubleFactorizationBlockEncoding.from_eigensystems(
    ((1.0, (diagonal, diagonal)), (-3.0, (-diagonal, diagonal))),
    (((2.0, (0.0, 1.0)),),),
    outer_bits=1, inner_bits=1, rank_rotation_bits=2, basis_rotation_bits=4,
)
```

The conversion follows the adjacent rotation chain in Eq. 57 of
[von Burg et al.](https://arxiv.org/pdf/2007.14460#page=71) and rounds the
half angles to the nearest fractional-turn word. It consumes already
factorized real eigenpairs. The locked Qualtran Hamiltonian factory raises
`NotImplementedError`.

The locked resource declaration retains its original reflection
ports and the outer success control. The executable model uses the paper
reflection and its nonzero-factor control.

The two-orbital fixture emits 450 T-family instructions and 81 measurements
with four scratch wires. Native controlled rotations used in the rank
preparation remain outside that T-family instruction count.

For a supported shape, obtain the locked analytical baseline directly from
the operation before binding it to any wires. The same estimate is available
from `source_double_factorization_block_encoding_cost` with dimensions alone.

```python
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

encoding = DoubleFactorizationBlockEncoding.from_eigenvalues(
    (1.0, 2.0, 3.0), ((1.0, -1.0, 2.0),) * 24, ((0, 0),) * 75,
    outer_bits=2, inner_bits=2, rank_rotation_bits=3, basis_rotation_bits=4,
)
report = estimate_ftqc_resources(encoding.source_resource_model())
assert report.logical.gates.total_t_count() == 2200
assert report.logical.peak_logical_qubits == 49
assert report.emitted is None
```

The analytical report uses the source's shared alias precision and automatic
QROAM block sizes. The source's automatic block selection requires at least
as many rows as target bits for forward lookups. This excludes some small
executable examples. Pass `emitted_ir=circuit.to_ir()` for the corresponding
FlagQuantum circuit to report its instruction counts separately.

| Same 24-factor input | T metric | Measurements | Width |
| --- | ---: | ---: | ---: |
| Locked analytical model | 2,200 T equivalents | 54 | 49 peak logical qubits |
| FlagQuantum emitted IR | 2,650 T/T† instructions | 603 | 82 declared qubits, including seven scratch wires |

The emitted IR also contains 36 native controlled rotations before synthesis.
