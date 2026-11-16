# Resource classification source-file audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Frozen source behavior | FlagQuantum owner |
| --- | --- |
| `_get_basic_bloq_classification`, `classify_bloq` | `classify_resource_family` recognizes typed arithmetic, rotations, conjunctions, swaps, T and Toffoli. Ordered custom rules return the first match, or `other`. Typed adjoints retain their family. |
| `classify_t_count_by_bloq_type` | `attribute_t_cost` composes standard filtering with user rewrites, cuts traversal at classified calls and multiplies modeled unit costs by leaf multiplicity. |
| `bloq_is_clifford`, `bloq_is_t_like`, `bloq_is_rotation` | `classify_resource_gate` retains the pinned finite angle and exponent tables, strict tolerance and single-control rule. |
| `bloq_is_state_or_effect` | `is_resource_state_or_effect` identifies all eight one-qubit X/Z states and effects. Measurements and identity remain outside the predicate. |

Q-0812 covers all 18 unchanged original cases. The native combined fixture
uses independent cost models for all seven original default examples, along
with the zero-T uniform preparation case. Model tests cover first-match order,
adjoint identity, the weighted bins, the classified cutoff and zero costs.

Q-0811 retains automatic default assignment for generic component summaries.
Qualtran derives families from source module paths. Native summaries currently
require an explicit family rule for preparation, lookup and reflection when
represented by generic `ComponentCost` keys. The audit fixture declares these
families using the cost objects themselves.

Classification and attribution emit zero primitive gates. Existing state
preparation and tensor-network effect tests cover physical basis vectors,
complex contraction and operand binding separately.
