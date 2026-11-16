# Recursive costing source-file audit

The reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Checklist | Source | Native contract and evidence |
| --- | --- | --- |
| Q-0802 | `resource_counting/_costing.py` | `CostMetric` owns the recursive rule, omission identity and optional static validation. `query_component_cost` uses a caller-owned cache for repeated visits. `CostMetric.cost_cache` gathers one metric's rows and `query_component_costs` collects the resulting metric table. Node and cost value types are independent, so symbolic costs use the same traversal. |
| Q-0803 | `resource_counting/_costing_test.py` | All four frozen tests run unchanged. Existing native tests compare call order, aliases, cache contents, seeded/static precedence and independent metric tables. `test_costing_file.py` checks both file hashes, symbolic metric rows, validation and composite recomputation. |

Generalization precedes cache lookup. Removing a node returns the metric's
identity. A seed for an ordinary node takes priority over static costs. The
metric validates static declarations only. A shared computed leaf is written
once and reused across subsequent visits within that metric.

The source bypasses cached composite values and omits computed composites
from the cache. Its static composite branch still validates and stores the
explicit value. The native `query_program_cost` adapter applies the same
policy to verified typed call decompositions. Primitive and mixed native
bodies retain ordinary memoization. Program queries use the verified
module's definition names and metric-provided callee selection. Native cycles
are rejected explicitly.

These queries operate on cost declarations and Program metadata and emit
zero primitive gates. Execution, primitive counts and work-wire ownership
remain covered by the existing normal append and hierarchical Program tests.
