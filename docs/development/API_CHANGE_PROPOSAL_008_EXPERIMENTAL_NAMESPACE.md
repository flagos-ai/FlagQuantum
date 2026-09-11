# API Change Proposal 008: A Smaller Experimental Namespace

## Status

**Governance baseline frozen — approved and frozen.**

- Target: first public alpha.
- Scope: public routing within `flagquantum.experimental` only.
- Stable root API changes: none.
- Feature implementations removed: none.
- Machine contract: `contracts/experimental-namespace-v1-candidate.json`.
- Implementation authorization: the API owner explicitly requested experimental
  namespace reduction on 2026-09-01.
- Approval: the API owner explicitly approved this governance baseline with
  `approve 008-010` on 2026-09-01.
- This freezes no second-level experimental feature and does not change the
  freeze status of Proposals 002-007.

## Problem

`fq.experimental` previously exposed 99 flat names mixing low-level distributed
records, dynamic circuits, third-party interop, TEBD, and multiple generations of
numerical experiments. Autocompletion did not reveal ownership, and maintainers
could not independently review, stabilize, or retire a domain.

## Decision

Expose only eight domain namespaces at the top level:

```text
fq.experimental
├── distributed
├── dynamic
├── execution
├── interop
├── mps
├── numerics
├── planning
└── simulation
```

Feature symbols move into their second-level namespaces instead of flat
`fq.experimental` exports. Implementations remain. This changes public routing
and repository callers without redesigning execution/training semantics.

Typical migrations:

```python
fq.experimental.dynamic.run_dynamic(...)
fq.experimental.distributed.train_distributed_statevector(...)
fq.experimental.interop.qiskit.from_qiskit(...)
fq.experimental.numerics.run_double_single_conformance(...)
fq.experimental.simulation.run_tebd(...)
```

## Lifecycle

Experimental APIs have no compatibility guarantee, but cannot remain indefinitely:

1. New entries require owners, scope, tests, and exit conditions.
2. Review at least every two minor versions or six months.
3. Within twelve months, stabilize, explicitly renew, refactor, or remove.
4. Stable promotion requires a separate proposal and semantic, error, interop,
   documentation, and real-environment gates.
5. Classification does not establish eligibility for stability.

## Acceptance Criteria

- [x] `fq.experimental.__all__` contains only eight domain namespaces.
- [x] Existing implementations remain accessible through explicit second-level namespaces.
- [x] Repository tests, examples, tools, and docs no longer use flat experimental entries.
- [x] Stable root `fq.__all__` remains unchanged.
- [x] Machine contracts fix the classification and disclaim compatibility guarantees.
- [x] API owner approved this classification as the first public alpha experimental governance baseline.

## Next Stage: Stabilization Review

Review each capability for user value, complete semantics, backend consistency,
maintenance ownership, and evidence maturity. Prioritize clear user workflows
with cross-version verification. Development evidence types, rank/shard records,
and hardware-specific primitives do not enter stable APIs by default. Every
promotion needs its own proposal; second-level placement does not imply stability.
