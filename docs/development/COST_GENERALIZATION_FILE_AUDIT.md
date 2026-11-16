# Cost generalizer composition source-file audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Checklist | Frozen source | Native owner and evidence |
| --- | --- | --- |
| Q-0804 | `resource_counting/_generalization.py` | `compose_cost_generalizer` in `_component_trace/call_graph.py` applies the rules in order, preserves identity with an empty chain, and short-circuits an omitted input or intermediate result. |
| Q-0805 | `resource_counting/_generalization_test.py` | The original case runs unchanged. `test_cost_generalization_file.py` compares both orders, each single rewrite, the empty chain, and the exact callback events for a tagged atom and an omitted input. |

The native entry accepts either a single callable or a sequence of callables.
The existing cost query tests verify generalization before lookup in the
metric cache. The weighted source graph test verifies that equivalent children
merge after rewriting and omitted children contribute no edge. These are
metadata transformations and emit zero primitive gates. Built-in rewrite
coverage is tracked separately with the public generalizers source family.
