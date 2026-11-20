# Bounded Cirq layer lookahead

Status: integrated in PR #1234 after complete PR CI passed. This change belongs
to Compiler's `single_device_fast_path`.
It adds no public API, dependency, result field or serialized schema.

## Source and adaptation

The source is Cirq 1.7.0, revision
`de651a7886d1a1383d595f95f36b93ed3ea91685`, specifically
[`RouteCQC._choose_optimal_swap`, `_choose_pair_of_swaps` and `_cost`](https://github.com/quantumlib/Cirq/blob/de651a7886d1a1383d595f95f36b93ed3ea91685/cirq-core/cirq/transformers/routing/route_circuit_cqc.py).
The adapted file retains the Cirq Developers copyright and Apache-2.0 notice.
No Cirq circuit, mapping manager or NetworkX graph enters production code.

The proposal filters candidate SWAPs lexicographically by maximum then total
interaction distance in each successive layer. If the bounded horizon leaves
single candidates tied, it tries disjoint SWAP pairs. Complete short horizons
retain the upstream deterministic tie choice. At exactly eight layers an
unresolved tie remains unresolved, matching the inspected upstream boundary.

FQ projects layers from its existing dependency DAG and pending predecessor
counts. Quantum, classical and barrier dependencies remain owned by that DAG.
Sparse local predecessor decrements avoid copying the full graph per proposal.
Scoring temporarily exchanges physical endpoints without mutating caller state.
This adapts upstream selection, not all of RouteCQC or its moment scheduler.

## Selection and limits

A direct replacement was rejected after development inputs regressed. The
existing deterministic route and three seeded routing trials stay available.
One additional trial uses layer proposals, returning to the existing heuristic
when no proposal is resolved. This is compiler search, not numerical execution
or backend substitution.

All five candidates are ranked by delivered SWAP cost, including final layout
restoration and, for directed devices, both SWAPs of each operand repair.
Only strictly cheaper candidates replace the incumbent. Equal costs retain the
previous selection. The original layout search trajectory remains unchanged.
Speculative SWAPs can still be undone by existing progress recovery, and only
committed prefixes permit cost pruning.

Per proposal the projection visits at most 160 operations and eight two-qubit
layers. Disjoint pair enumeration permits at most 32 candidate edges, hence at
most 496 pairs; larger ambiguous candidate sets use the existing heuristic.
Single-candidate scoring still depends on the frontier and device degree.
These are search-count bounds, not elapsed-time or memory guarantees. Extra
trials cost compilation time. The caller must still impose resource limits.

## Reproduction and evidence

```bash
python -m benchmarks.compiler_layer_lookahead > /tmp/layer-lookahead.json
python -m pytest tests/unit/test_compilation_layer_lookahead.py \
  tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_sabre_routing.py -q
```

The optional source-conformance test calls the installed Cirq implementation
through a small mapping adapter on 300 dependency graphs. It passed with Cirq
1.7.0. Environments without Cirq skip that cross-project test; native acceptance
remains mandatory. Other tests cover a genuine disjoint-pair decision, the
exact eight-layer boundary, unresolved predecessor joins, input immutability,
projection and pair budgets, full operators and delivered-cost nonregression.

[Raw development rows](evidence/cirq-layer-lookahead-development.json) retain
all 80 cases, source hashes, environment and diagnostic timing: widths 6, 8, 12
and 16, line/grid topologies, ten deterministic seeds, and four times the width
in CX instructions. All three alternatives pass exact full GF(2) operator,
restored-layout and coupling checks.

| Configuration | Total delivered SWAPs | Sum of logical two-qubit depths |
| --- | ---: | ---: |
| Existing four trials | 6768 | 4628 |
| Standalone layer proposal | 7364 | 4889 |
| Five-trial selection | 6694 | 4576 |

Seventeen cases improve under selection and 63 tie. The standalone proposal
has worse aggregate counts; that negative result is deliberately retained.
These are development inputs, not prospective qualification or an SDK ranking.
Logical SWAP depth is not target-CZ depth. One timing sample on a shared host
without warmup establishes neither speedup nor compliance with resource budgets.
No peak-memory or hardware result is claimed. Phase-sensitive native routing
tests complement this CX-only probe; the probe itself does not test phases.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(8).h(0).cx(0, 7).ry(3, theta=0.2).cx(3, 6)
compiled = compile(source, coupling_map=CouplingMap.grid(2, 4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

The ordinary automatic selector can use the improved SABRE candidate. The
observable contract stays restored output layout. More search does not prove
a globally minimum CZ count, depth, duration or error. The integrated upstream
adaptations and remaining gaps are listed in [the audit](COMPILER_UPSTREAM_REUSE.md).

## Validation status

Local verification includes 1,449 compiler unit and integration tests, including
the optional Cirq source comparison; strict mypy over 1,134 source files; scoped
Ruff; architecture, Stable Core API, ownership, English-content and local-link
checks. The broad parent smoke/unit run previously reported failures, including
evidence-revision provenance diagnostics, and is not represented as passing.
PR CI passed on `71614dd3935b2c0adfb9336d306e44706f461762`: 239 checks
passed and three optional checks were skipped. The implementation was integrated
as `858df9dbd5ba964e8e59b3383304547710a1dee5`, whose installed-wheel smoke
also passed. This does not convert the earlier incomplete local run into passing
evidence. Prospective resource-qualified SDKBench execution and independent
replay remain required before comparative claims.
