# Typed resource call-graph factory

## Ownership and use

`export_resource_call_graph_dot` in Drawer composes existing Compiler typed
resource recipes, full recursive QEC costs and caller-provided peak widths.
The normal `ComponentCost` exporter remains the concrete model workflow.
The typed entry handles symbolic multiplicities, automatic compound recipes and
explicit declaration rows. It produces DOT for the existing bounded SVG renderer.

```python
from flagquantum.compiler._component_trace.resource_generalization import ControlledSwapCall
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

root = ControlledSwapCall(3, approximate=True)
dot = export_resource_call_graph_dot(
    root,
    logical_width=lambda node: 7 if node == root else None,
    max_depth=0,
)
assert '<TD>Qubits</TD><TD>7</TD>' in dot
assert '<TD>Ts</TD><TD>12</TD>' in dot
```

The callback owns width estimation and is called once for every node in the
complete resource graph. It can use `estimate_connection_width`, a previously
computed native register-program width, or a width table. It returns `None`
when a metric did not visit that node. Zero remains a calculated width.
A missing table key propagates `KeyError`. A retry builds fresh local caches.
Declared calls, width tables and all quantum state remain caller-owned.
Display depth limits graph visibility while full costs and width callbacks
continue to evaluate hidden nodes. Full-graph cycles fail even at depth zero.

## Locked source chain and adaptations

The comparison uses unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Four source Git blobs are checked
against the fixed inventory, covering the drawing factory, qubit estimator,
MultiAnd and CSwapApprox. The native chain follows `from_bloq`, graph generation,
full cost queries, sparse cost-row formatting and DOT/SVG rendering. Compiler
keeps resource identities, recipes and width estimation. Drawer composes them.

The source width query and QEC query can visit different nodes. For negative
MultiAnd, X has gate cost and no width row. Concrete CSwapApprox decomposes for
width accounting, while its category graph uses arbitrary Clifford leaves.
Symbolic CSwapApprox uses the source signature/callee width fallback. Tests
preserve these sparse rows explicitly instead of adding inferred width labels.
The native callback makes the existing width owner's boundary explicit.
Automatic inference across every typed resource and foreign graph remains open.
Custom titles use the existing callback. Native class names, metric identities,
error wording and HTML table presentation remain documented adaptations.
Arbitrary subclass node-property overrides retain their separate open scope.

The original seven source test functions and helper remain unchanged. Their
native execution now uses this production factory for QEC queries and table
formatting. The fixture-specific zero-bystander width computation remains at
the test boundary. The previous duplicated test-only gate annotation loop was
removed. Existing source HTML presentation adaptation still operates after DOT.

## Acceptance

68 new cases plus the existing 23 source-asset cases pass together in 4.84
seconds. Sixty cases compare complete visible cost rows and edge labels for
positive/negative MultiAnd and concrete/symbolic CSwapApprox across five
aggregation modes and three depths. Every case parses actual Graphviz SVG.
Further cases exercise connection-width bystanders, symbolic declared DAGs,
hidden width failures and fresh retry, invalid depths and cycles.

1390 concentrated cases pass in 34.77 seconds, including existing connection
width, exact Fredkin, measured adjoint, classical parity and source-asset tests.
87 independent real-kernel notebook/CPU/TN integration cases pass in 13.28
seconds. There are zero skips. Concentrated and integration runs expose 1112
and two existing warnings respectively. Strict typing covers three changed
Python files. Black, Ruff, Compiler scope, architecture, internal IR and protected
API checks pass. The native Kickmix extension and pinned dependency versions
are checked before regression. The module has seven passing doctest examples,
and the Markdown workflow executes successfully.

Measured example/Drawer coverage is 100/95 percent, 96 percent combined.
Remaining uncovered lines belong to existing Markdown edge-label and renderer
failure paths. Initial diagnostics exposed sparse width-row differences,
postponed symbolic annotations, test union narrowing, pydot's external typing
and symbolic pretty-print edge labels. They were corrected without changing
source files or expected source behavior. Diagnostic logs remain in the batch
reference environment.

| Resource evidence | Value |
| --- | --- |
| Drawing added primitive gates / work wires / measurements | 0 / 0 / 0 |
| Three-bit approximate swap declared categories | T12, Clifford65 |
| Symbolic approximate swap declared categories | T4*n, Clifford22*n - 1 |
| Exact Fredkin emitted primitives | CX8, H2, T4, Tdg3, work0, measurement0 |
| Exact Fredkin terminal-T / default QEC estimate | 7T / 4T |
| Existing measured merged-Fredkin emitted T-equivalent / measurement | 11 / 1 |
| Merged model components | 4 + 7 + 0 |

Existing gate-sequence, full complex-amplitude, measurement-branch, work-zero
and `circuit.append(op)` checks pass in the concentrated suite. Resource graph
formatting uses declarations without numerical quantum execution.

Q-0734.04 records this capability. Q-0734 stays open for universal width
admission and arbitrary property overrides. Foreign admission, CDDL renderer
packaging, browser interaction, historical QFT 190 versus 496 and the historical
full-default 600-second timeout retain their independent open records.
