# Recursive logical width queries

Compiler owns `query_logical_width` in
`flagquantum/compiler/_component_trace/logical_width.py`. It reuses the existing
cost-query cache, generalizer and cycle detection. A decomposition owner computes
its connected peak using the recursive child query and the existing
`estimate_connection_width` or register-program estimator. Drawer consumes the
resulting sparse cache through its existing width-provider callback.

## Example

```python
from flagquantum.compiler._component_trace.logical_width import query_logical_width
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.resource_generalization import ResourceCall
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

root = ComponentCost("entry", 3)
widths: dict[ResourceCall, SymbolicInt] = {}
query_logical_width(
    root,
    signature_width=lambda node: 3,
    callees=lambda node: (),
    cache=widths,
)
dot = export_resource_call_graph_dot(root, logical_width=widths.get)
assert widths == {root: 3}
assert "entry" in dot and "3" in dot
```

## Source behavior and ownership

The locked reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/resource_counting/_qubit_counts.py`, `QubitCount.compute`.
Its decomposition path delegates liveness to `_cbloq_max_width`. FlagQuantum
reuses the previously accepted connection-width implementation for that work.

A decomposition result takes precedence, including zero. Returning `None`
explicitly admits the fallback maximum of signature and recursive callee widths.
That fallback can underestimate simultaneously live workspace. Every fallback
callee is visited even with zero repetitions, matching the original source.
Call multiplicity does not multiply a peak width.

Only visited nodes enter the width cache. Hidden decomposition children may have
entries while visible gate-category children have none. `cache.get` preserves
this distinction for the existing typed resource drawing factory. Cache and
static overrides follow the existing cost-query precedence. Reusing a cache
requires matching decomposition owners, signature rules, declarations,
generalizers and overrides. Computation failures propagate and retain completed
child entries. Failed parents receive no cached value.

Source-specific decomposition exceptions are translated to `None` at the adapter
boundary. Production Compiler imports no Qualtran objects. This batch supplies
the recursive query mechanism. Universal automatic binding from every typed
resource key to its signature and native register decomposition remains open
under Q-0734. No new serialized graph or register schema is introduced.

## Acceptance evidence

`tests/unit/test_logical_width_query.py` compares complete recursive width caches
with the unchanged source for twelve cases, including symbolic/interior
allocation, independent/serial components, three many-allocation strategies,
multiple allocation, both MultiAnd polarities and concrete/symbolic CSwapApprox.
The test adapter translates source connections to existing native width steps.
It supplies no source-computed local widths to the native recursion. The frozen
source Git blob is checked against the immutable inventory.

Native tests verify a child peak of 8 plus 5 bystander qubits gives 13, sparse
cache delivery to the production drawing factory, fallback zero multiplicities,
symbolic maxima, authoritative zero, cache/static/generalizer precedence,
cycles, retries, invalid computed widths and unexpected decomposition failure.

The query emits no circuit operations. Added primitive gates, work qubits and
measurements are each zero. Numerical complex amplitudes, measurement branches,
workspace cleanup and `circuit.append(op)` remain covered by the existing
execution regression. Executed primitive counts and analytical resource models
retain separate ledgers.
