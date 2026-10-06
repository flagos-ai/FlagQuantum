# API Change Proposal 068: Let a routing plan describe a placement it does not restore

## Status

**Proposed. No Stable Core change requested. Not implemented by this proposal.**

This proposal asks for a decision, not for a code change. It adds no strategy, no
pass, and no public name. It records what the deployment routing contract
(`contracts/` has no v1 file for it; the schema literal lives in
`flagquantum/deployment/routing_evidence.py`) refuses today, with a measurement
for each refusal, and proposes the smallest vocabulary that would let a caller
describe the programs the compiler already produces.

The Compiler owns `flagquantum/compiler/**`. The contract below is consumed by
`flagquantum/deployment/routing_evidence.py` and
`flagquantum/deployment/cloud.py`, which are Remote-owned. Accepting this
proposal is therefore an Integration decision taken with the Compiler and Remote
maintainers, and the implementation, if approved, is a Remote change that the
Compiler supplies inputs to.

## Problem

`flagquantum/compiler/` today offers four routing strategies and two layout
planners. Every one of them can produce a program, and some of them can produce a
program that no deployment package can carry. The deployment contract is not
wrong about the plans it accepts; it is silent about the plans it refuses, and a
caller has no way to tell "this program is deployable" from "this program is
deployable only if you happened to route it the one way the contract knows".

The contract's job, in its own words, is to *"validate and normalize a routing
plan for provider serialization"*. Serialization is not the problem: a plan
carrying a placement serializes exactly as well as a plan carrying the identity.
The problem is that the contract answers a narrower question than the compiler
asks, and the difference is paid as a refusal at the last step before submission.

### The v1 plan in one paragraph

`flagquantum_routing_plan_v1` is emitted by
`flagquantum/compiler/routing.py::_routing_metadata`. It states the strategy, the
device as a coupling map, three permutations — `initial_logical_to_physical`,
`pre_restore_logical_to_physical`, `final_logical_to_physical` — a boolean
`mapping_restored`, a direction-semantics string, four counts, and a path-cache
delta. It always writes `final_logical_to_physical` as the identity and
`mapping_restored` as `True`, because every strategy in `ROUTING_STRATEGIES`
walks each logical wire back to its own physical wire before returning.

The second plan, `flagquantum_directed_routing_plan_v2`, is emitted by
`flagquantum/compiler/directed_topology.py`. It describes routing on a
`DirectedCouplingMap` and it reports the placement it started from and the
placement it ended on, which are not the identity when the caller asked for one.

## Measurement

Four programs per topology per strategy, on four topologies (`line4`, `ring5`,
`line6`, `grid3x3`), through the public entries `flagquantum.compiler.compile`,
`flagquantum.compiler.plan_dense_layout`,
`flagquantum.compiler.directed_topology.route_to_directed_topology` and
`flagquantum.compiler.layout.remove_layout_restore`, validated by
`flagquantum.deployment.routing_evidence.validate_deployment_routing_plan`.
48 routed plans, 12 reduced plans, 9 placement-routed plans, and one program
routed against a re-ordered serialization of the same device.

The refusals are layered, and the layers are measured rather than assumed: a
`sabre` plan is refused by the strategy clause, and `sabre_layout` is refused by
the strategy clause *and* the initial-permutation clause; a placement-routed plan
is refused by the schema clause before any permutation is inspected. Each obstacle
below states what refuses first on its own reproduction, and
`tests/unit/test_routing_plan_v2_candidate.py` re-derives the next refusal by
rebuilding the validator's own source with one clause replaced — an experiment,
not a reading of the source.

### Obstacle 1: the strategy vocabulary is a copied literal

| strategy | verdict | count |
|---|---|---|
| `restore_after_each_gate` | accepted | 12 |
| `persistent_layout` | accepted | 12 |
| `sabre` | `unsupported routing strategy` | 12 |
| `sabre_layout` | `unsupported routing strategy` | 12 |

The validator holds a two-element set literal:

```python
if strategy not in {"restore_after_each_gate", "persistent_layout"}:
```

`flagquantum/compiler/routing.py:24` holds `ROUTING_STRATEGIES`, a four-element
tuple, and the compiler dispatches on it. Widening only that literal — every
other clause left in place — makes the `sabre` plan pass every remaining check on
the same programs; see `tests/unit/test_routing_plan_v2_candidate.py`. So this
obstacle is one copied literal, and its cost is that half the compiler's routing
strategies cannot reach a provider.

This is the same defect class the Compiler fixed for itself in `optimization_levels`:
a normative set copied into a consumer, so that adding a member to the authority
silently narrows the consumer instead of widening it. The proposal asks the
consumer to read the authority rather than a copy of it.

### Obstacle 2: the initial permutation must be the identity

```python
expected_identity = tuple(range(int(n_qubits)))
if initial != expected_identity:
    raise DeploymentRoutingEvidenceError(
        "routing initial permutation must be identity"
    )
```

Two of the compiler's own facilities produce a non-identity initial placement:

- `flagquantum.compiler.plan_dense_layout(circuit, coupling_map)` returns a
  placement tuple, documented in `capability-maturity.toml` as the planner "that
  can help on a device with unequal connectivity and the only one of the two that
  can reach a program the lowest qubits cannot hold at all";
- `sabre_layout` routes from a searched initial placement.

On `grid3x3` with a 9-wire program, `plan_dense_layout` returns `(0, 1, 3, 4, …)`
rather than `(0, 1, 2, 3, …)`. Six of the nine placement-routed plans measured
here start from a non-identity placement, and none of them can be described.

There is no field in the v1 plan in which the placement could be reported. The
refusal is not a judgement about the placement; it is the absence of a place to
put it.

### Obstacle 3: the final permutation must be the identity

| program | verdict | count |
|---|---|---|
| `restore_after_each_gate`, as routed | accepted | 12 |
| `restore_after_each_gate`, after `remove_layout_restore` | accepted (identity, nothing to remove) | 12 |
| `persistent_layout`, as routed | accepted | 12 |
| `persistent_layout`, after `remove_layout_restore` | `deployment routing must restore the final logical permutation` | 12 |
| `sabre`, after `remove_layout_restore` | `unsupported routing strategy` (obstacle 1) | 12 |
| `sabre_layout`, after `remove_layout_restore` | `unsupported routing strategy` (obstacle 1) | 12 |

`remove_layout_restore` is a public Compiler transformation whose entire purpose
is to *not* restore, and it says so in its own docstring: its result "is a program
to run against a local simulator or a provider that accepts per-physical-qubit
results, not a routing result to hand back as deployment evidence".

That sentence is honest and it is also the statement of the gap. "A provider that
accepts per-physical-qubit results" is a real provider class, and a program for
one of them is refused before the seam where a provider would be consulted. The
v1 plan can say `mapping_restored: true` and cannot say anything else.

The cost is not hypothetical: on the 140-program routing basis that
`benchmarks/compiler_routing_quality.py` uses, removing the restore phase halves
the `persistent_layout` SWAP count, and on the batch measured for W7-14 it turned
`persistent_layout` 11778 → 5889 and `sabre_layout` 5155 → 2854. Those programs
are runnable and they are not deployable.

### Obstacle 4: exactly one schema literal is accepted

```python
if plan.get("schema") != "flagquantum_routing_plan_v1":
    raise DeploymentRoutingEvidenceError("unsupported routing plan schema")
```

All nine placement-routed plans measured here are refused by this clause, before
any permutation is inspected. The directed path emits
`flagquantum_directed_routing_plan_v1` (3 of 9) or
`flagquantum_directed_routing_plan_v2` (6 of 9) depending on whether physical
workspace was allocated.

So the two plans that describe the same physical device, produced by two layers
of the same compiler, do not share a schema and cannot be compared. A consumer
that wants to know "is this program routed onto this device" has to know which
compiler entry produced it.

### Obstacle 5: the device identity is compared by edge order, in two places

`CouplingMap.edges` stores the caller's tuple verbatim; it is not canonicalized.
So two objects describing the same physical couplings compare unequal when their
edge lists were written in different orders — and `DirectedCouplingMap`
canonicalizes its edges while `CouplingMap` does not, so the two compilers' own
device types disagree about the same device by construction:

```text
DirectedCouplingMap(6, grid6): ((0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5))
CouplingMap(6, grid6):         ((0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5))
```

Both deployment sites test that ordered tuple:

```python
# flagquantum/deployment/routing_evidence.py
if coupling_map is not None and plan_edges != coupling_map.edges:
    raise DeploymentRoutingEvidenceError(
        "routing coupling edges do not match the deployment backend"
    )

# flagquantum/deployment/cloud.py
and tuple(existing_routing.get("coupling_edges", ()))
== backend.coupling_map.edges
```

The minimal reproduction needs no placement, no workspace and no foreign schema.
Route a six-wire program declared by `coupling_map=CouplingMap(6, edges)` where
`edges` is the same grid written in a different order, then deploy it against a
backend built from the grid in its canonical order. The plan is a v1 plan and it
says every permutation the v1 contract requires:

```text
v1 plan schema                = flagquantum_routing_plan_v1
plan coupling_edges           = ((1, 2), (0, 1), (0, 3), (2, 5), (1, 4), (4, 5), (3, 4))
backend coupling_edges        = ((0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5))
equal as tuples (the test)    = False
equal as sets (the meaning)   = True
```

Two refusals then follow from the one inequality:

- the plan is refused outright against the backend device with *"routing coupling
  edges do not match the deployment backend"*, even though the two devices are the
  same device; and
- through `create_deployment_package` the plan is treated as unrouted, the program
  is routed a second time, and the package is refused with *"post-optimization SWAP
  count must be within planned count"*.

The second refusal needs its own explanation, because it is a different bug: the
second pass reports `planned_inserted_swap_count = 0` while
`record_post_routing_optimization` counts `4` — the first pass's SWAPs, which the
second pass never planned. The invariant `retained <= planned` is sound for one
self-contained plan and false for a program that arrives already routed, so even
after the order comparison is fixed, re-routing an already-routed program has no
defined meaning. The proposal fixes the comparison and asks for the second
meaning to be decided rather than assumed.

The order comparison is the trigger; the count invariant is the amplifier. Together
they turn a routed program into a refusal whose message names a SWAP count rather
than the reuse decision that caused it.

This is the second order-sensitivity defect found in this device vocabulary: round
22 fixed `CouplingMap.shortest_path` returning a path that depended on which query
came first, and this is the same shape — an unordered fact compared as though its
serialization order were part of it.

## Decision requested

Admit a routing plan that reports a placement instead of asserting the identity,
in three moves.

### 1. Read the strategy vocabulary from its authority

Replace the two-element set literal with a membership test reading
`flagquantum.compiler.routing.ROUTING_STRATEGIES`. Adding a strategy to the
compiler then widens the deployment contract instead of silently narrowing it,
and the two cannot drift.

### 2. Report the placement in all three positions, with a named final state

Keep `initial_logical_to_physical` and `pre_restore_logical_to_physical` as they
are: a complete permutation of the logical wires, as a tuple of physical slots.
Replace the `mapping_restored: bool` companion of
`final_logical_to_physical` with a value from a closed vocabulary, so that a plan
states which of three things happened rather than whether one of two did:

| value | meaning | today's form |
|---|---|---|
| `identity` | every logical wire is back on its own physical slot | `mapping_restored: true` |
| `explicit_permutation` | the program ends on a named layout, reported in `final_logical_to_physical` | not expressible |
| `logical_result_physical_slots` | the program allocates device workspace and the results live on named slots | only in the directed v2 plan |

`mapping_restored: true` keeps its meaning: it is `final_layout == "identity"`. A
v1 plan that restores is a v2 plan with the final layout set to `identity`, so the
change is additive for every plan the contract accepts today.

The vocabulary is a string, not a boolean, for the same reason
`capability-maturity.toml` uses a level name rather than a number: a third state
cannot be added to a boolean later without changing what `false` means.

### 3. Compare devices as the set they are

Compare a normalized `frozenset` of edges rather than the edge tuple, in both
sites. The edge *order* is a serialization detail of whichever device object was
constructed, and the deployment question is whether the two describe the same
physical couplings.

This one is independent of the other two and can be taken alone: it is a
correctness fix, not a vocabulary change. Without it the deployment contract
answers "is this device object the one my plan was built from" when the question
is "is this the same device", and a plan can be refused against the very device it
was routed onto.

It does not, on its own, make re-routing an already-routed program correct. Once
the comparison is fixed, the programs measured here are no longer re-routed and
the count refusal stops being reachable *through this path* — but re-routing must
still be either defined or refused by name, because the second pass reports
`planned_inserted_swap_count = 0` for a program that already contains SWAPs. That
half is a decision, not a fix, and this proposal requests it rather than assumes a
value for it.

## Compatibility

- **No Stable Core change.** `docs/public_api_v1.json` is untouched. The routing
  plan is `metadata["routing"]`, not an export.
- **No `flagquantum.compiler` public signature change.**
  `route_to_topology`, `compile`, `optimize`, `route_to_directed_topology`,
  `remove_layout_restore` and the two planners keep their signatures.
- **Additive for every plan accepted today.** Both currently-accepted strategies
  write the identity, which becomes `final_layout = "identity"`.
- **`coupling_map` is not changed.** `contracts/directional-topology-layout-v2-candidate.json`
  pins `coupling_map_changed: false`, `public_root_export: false` and
  `default_path_change: false`; this proposal does not request any of them.
- **No default change.** The default routing strategy stays
  `restore_after_each_gate`. Nothing here makes the compiler produce a
  non-restored program by default.
- **The directed v2 plan and the routing plan stay separate documents.** Merging
  them is a larger change than this gap needs and is explicitly out of scope.

## Acceptance

If approved, the change is accepted when all of the following hold, and
`contracts/routing-plan-v2-candidate.json` records which of them is which:

1. A plan produced by each of the four `ROUTING_STRATEGIES` passes
   `validate_deployment_routing_plan` on a device it was routed onto.
2. A plan produced by `remove_layout_restore` passes when it names its final
   permutation, and is refused by name — not by a SWAP count — when it does not.
3. A plan produced from an explicit `plan_dense_layout` placement passes, with
   the placement readable from the plan alone.
4. A plan routed against one serialization of a device validates against another
   serialization of the same device, in both comparison sites, and the reuse
   decision is observable in the package metadata rather than only in the SWAP
   counts that follow from it.
5. Re-routing an already-routed program is either refused by a message that names
   the reuse decision, or defined with both passes' counts reported separately.
   It is never reported as one pass that planned nothing and retained something.
6. Every clause of `validate_deployment_routing_plan` that refuses remains
   reachable by some input, so that the widened contract is still a contract.
7. `tests/unit/test_routing_plan_v2_candidate.py` reconciles this document's
   obstacle list against the running code, so that closing one obstacle without
   updating this proposal fails a test rather than leaving a stale claim.

## Non-goals

- **A fifth routing strategy.** This is about describing the programs the
  compiler already produces.
- **Changing which layout the compiler restores to.** `restore_after_each_gate`
  and `persistent_layout` keep restoring; the proposal adds a way to say what
  happened, not a way to skip.
- **Making `remove_layout_restore` deployment evidence.** Its own docstring says
  it is not, and that stays true: what changes is that a provider which accepts
  per-physical-qubit results can be handed the program with a plan that describes
  it, instead of the caller having to strip the metadata first.
- **Noise- or calibration-aware placement, physical ancilla allocation, partial
  layout, or observable remapping.** All four are listed as excluded claims in
  `contracts/directional-topology-layout-v2-candidate.json` and none is
  requested here.
- **A Qiskit-parity claim.** This is a contract gap in FlagQuantum's own
  deployment seam. The parity matrix records no entry for it, and one is not
  requested.
- **Capability maturity.** No capability level changes. This proposal adds no
  capability to `capability-maturity.toml`.
