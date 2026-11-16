# Success probability source-file audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Checklist | Frozen files | Native owner and evidence |
| --- | --- | --- |
| Q-0808 | `resource_counting/_success_prob.py` | `_component_trace/cost_query.py` evaluates the recursive probability product with multiplicative identity 1.0. It uses the shared query for generalization, seeded cache precedence and static overrides. The frozen display label is `success prob`. |
| Q-0809 | `resource_counting/_success_prob_test.py` | `test_success_probability_file.py` checks both frozen blobs, the original coin-flip case, and the complete result cache with the existing concrete and declared probability queries. |

Four calls to a component with static probability 0.5 yield 0.0625. The completed
cache holds `CoinFlip = 0.5` and `Algo = 0.0625`. The source callback returns
`1.0*0.5**n` for a symbolic repetition despite its float annotation. The
declared query retains that expression and matches the source cache, including
zero-multiplicity children. The existing `ComponentCost` path retains its
integer repetition contract and omits zero-multiplicity calls from its cache.

The probability declaration emits zero primitive gates. The independent
execution checks in `test_component_success_probability.py` exercise the
measurement-assisted AND erase through normal append. Both outcomes have
weight 1/2, restore the coherent state and clear the work wire. The unconditional
operation has modeled success 1.0.
