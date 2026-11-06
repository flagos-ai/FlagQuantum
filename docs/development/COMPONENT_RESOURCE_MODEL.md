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

This counts a shape-only program under the source gate model. Peak qubits
depend on register lifetimes and use the separate width model. Automatic
scheduling and physical error-correction estimates remain separate.

- [Qualtran v0.7.0 gate costs](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py)
- [Qualtran v0.7.0 logical width](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_qubit_counts.py)
- [Qualtran v0.7.0 success probability](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_success_prob.py)
- [Qualtran v0.7.0 call graph](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_call_graph.py)
- [Qualtran v0.7.0 cost queries](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_costing.py)
- [Qualtran v0.7.0 symbolic mathematics](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/symbolics/math_funcs.py)
