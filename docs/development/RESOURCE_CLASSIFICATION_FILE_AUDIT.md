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

Q-0811 also covers default classification of native component summaries.
The model owns a semantic family from the same ten-family vocabulary as
the frozen table. Arithmetic, rotations, lookup, preparation, reflection,
multiplexers and bounded swaps declare their family at their model owner.
Typed gate keys continue to supply their own classification. Generic summaries
without a declared family use `other`. The native default does not infer
family from display text or Python modules. Explicit custom rules still replace
the default with the first matching rule. The original fixture now uses the
default native classifier throughout. Additional tests cover same-name summaries with distinct families.

Classification and attribution emit zero primitive gates. Existing state
preparation and tensor-network effect tests cover physical basis vectors,
complex contraction and operand binding separately.
