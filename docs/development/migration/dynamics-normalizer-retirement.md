# Dynamics normalizer retirement

Dynamics subset of the complete Simulation private-binding retirement,
implemented after PR1335 head `f986fb4026b46a585a165becaaae52f8f3bce6cc`
qualified with 243 successful checks and three workflow-conditional skips.

The evolution engine previously retained fifteen unused private imports for
older callers and tests. Those bindings are removed; their existing definitions stay
in `simulation/dynamics/problem.py`. Four test consumers use that authoritative
module directly, preserving all numerical assertions. The engine keeps its
actual implementation dependencies and no private compatibility wrapper.

Six other unused, undeclared imports have no current consumers: `Iterable`,
`Observable`, `ValidationError`, `PauliSum`, `PauliSumTerm` and `SOLVE_RESTART`.
They are removed from the internal engine namespace; their actual standard
library, Core, error, Matrix Free and integrator definitions remain unchanged.
They are absent from the maintained historical facade's declared exports.

The seven declared public evolution exports remain: `CollapseOperator`,
`EvolutionPlan`, `EvolutionResult`, `EvolutionValidationError`,
`amplitude_damping`, `evolve_density_matrix` and `plan_density_matrix_evolution`.
Their historical import facade, field hints, function/class identities,
serialized payloads and existing itemized compatibility basis are unchanged.
Private helpers are not frozen public APIs under
[the public API protection policy](../PUBLIC_API_PROTECTION.md).

The cumulative ledger for the complete Simulation batch has 545 rows;
the public-entry inventory remains 102 with no additions.
[The complete acceptance record](simulation-private-binding-acceptance.md)
records formal regressions, incoming consumer obligations, AST invariance and
installed public workflows/serialization. The child requires its own exact-head
CI before qualification.

The ledger row retires imported aliases and claims no moved definitions.
Existing definition-source identities remain attached to the original
Problem migration; the checker and frozen vocabulary records are unchanged.
