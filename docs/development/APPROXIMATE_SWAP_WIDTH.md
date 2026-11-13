# Approximate controlled-swap and fanout width bindings

The automatic Compiler width query delegates fanout to
`registers/bits/fanout_width.py` and approximate exchange to
`registers/primitives/relative_controlled_swap_width.py`. Both use the shared
`registers/resources/through_call_width.py` projection into the existing
connection scheduler. Each listed call consumes and returns the same wire set.
The projection keeps idle wires alive and queries child peaks in connection
order. Allocation and non-through outputs retain their separate owners.

The approximate exchange emitter and width owner share the literal instruction
segments around fanout. Source component boundaries retain split/join word
views and a distinct fanout call. Adjoint reverses the existing literal stream
and retains the fanout direction in its resource key. Concrete positive Python
integers decompose. Sympy integers and symbolic widths preserve source fallback
and sparse cache visitation. Source zero-width decomposition raises RuntimeError.
Native admission raises ValueError. The fallback can underestimate workspace.

```python
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledSwapCall, FanoutCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

swap = ControlledSwapCall(3, approximate=True, adjoint=True)
assert query_resource_logical_width(swap) == 7
assert query_resource_logical_width(FanoutCall(3, True)) == 4
assert '<TD>Qubits</TD><TD>7</TD>' in export_resource_call_graph_dot(swap)
```

## Resource accounting

Qualtran v0.7.0 counts forward MultiTargetCNOT as one Clifford in modern QEC
mode. Its generic Adjoint is decomposed into `2*n-1` CNOTs. Legacy QEC mode
uses that decomposition in both directions. Terminal T cost is zero in both
directions. Width and category caches can therefore visit different children.
Drawer retains category rows outside the displayed terminal graph safely.

For a three-bit approximate exchange, the source category recipe estimates
12 T and 65 Clifford operations. The concrete primitive stream is checked
separately against the unchanged source decomposition. This width feature adds
zero physical gates, work qubits and measurements. Execution acceptance retains
the existing complex-amplitude, clean-workspace and append workflows.

## Acceptance scope

The source is locked to Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Three frozen source blobs are
verified. Complete cache values and insertion order cover both directions,
widths one, two, three, five and eight, Sympy integers and symbolic widths.
Ninety full-DOT comparisons cover source-computed width tables across five
aggregations and three visibility depths. Seeded child peaks verify bystanders.
Both modern and legacy category models, zero-width failure and retry, and
invalid wire bindings are checked independently.

Arithmetic and general register-program automatic width bindings remain open
under Q-0734. The source reference is unchanged.

Local acceptance passed 194 focused tests in 7.40 seconds, 1,957 concentrated
regression tests in 49.91 seconds and 87 integration tests in 13.75 seconds,
with no skips. The concentrated run reports 1,112 upstream deprecation warnings.
The integration run reports two existing deprecation warnings. Strict typing
covers all 11 changed Python files. Black, Ruff, team scope, architecture,
internal IR and protected API checks passed. Nine Drawer doctests and the
Markdown example passed. The three new owners have 100 statements and 50
branches, all covered. The shared resource width dispatcher is fully covered.
The measured seven-module coverage scope is 98 percent overall.

For width three, the actual primitive counts are CX17, H24, S12, Sdg12, T6
and Tdg6, with zero work and measurements. The QEC model separately reports
T12 and Clifford65. The reference comparison executes both resource queries
and the original source primitive stream. Initial tests exposed the fanout
adjoint category mismatch and the disjoint cost-cache drawing failure. Both
are retained as regression cases.
