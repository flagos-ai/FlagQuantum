# Automatic routing as the native compile default

Status: implementation authorized by the user on 2026-11-18. Authorization
covers the exact default change below; it does not declare implementation,
release qualification, or CODEOWNER review complete.
Owner: Compiler, with API-owner review for the protected default.
Classification: `single_device_fast_path`.

## Problem and user journey

An ordinary caller supplying a topology currently receives
`restore_after_each_gate`, even though the repository already has deployable
SABRE and an automatic strategy selector. Explicit expert configuration can
therefore materially change routing cost. The default should use the existing
selector rather than require users to discover that configuration.

The selector currently ranks admissible plans by expected link faults when
calibrated, then inserted SWAP count, then instruction count, with deterministic
candidate-order ties. This proposal preserves that ranking. It does not rename
minimum SWAP count as minimum CZ cost or minimum runtime.

## Exact proposed contract change

In `flagquantum.compiler.pipeline.compile`, re-exported as
`flagquantum.compiler.compile`, change only:

```python
# Existing
routing_strategy: str = "restore_after_each_gate"
# Proposed
routing_strategy: str = "auto"
```

Parameter order, accepted values, return type, and other defaults remain the
same. Calls with an explicit strategy retain their existing behavior. Calls
without a topology perform no route search. `route_to_topology` remains the
explicit strategy primitive with its existing default.

When topology is supplied and strategy omitted, the existing selector chooses
among `restore_after_each_gate`, `persistent_layout`, and `sabre` when the device
allows them. Calibrated devices continue excluding strategies that ignore link
weights. `sabre_layout` remains opt-in because nonidentity initial layouts are
not admitted by the existing deployment contract. Identity final restoration
and all existing direction, topology, annotation, and failure contracts remain
in force. No free-final-layout change is bundled into this proposal.

## Compatibility and migration

This is a behavioral and signature-default change. Instruction ordering, SWAP
count, compile cost, program hashes, routing metadata, and exported text may
change for callers relying on the old implicit policy. The existing
`routing_strategy_selection` summary appears on newly automatic calls; its
schema is not expanded. Numerical and layout semantics must remain unchanged.

To preserve the earlier policy explicitly:

```python
import flagquantum as fq
from flagquantum.compiler import compile

circuit = fq.Circuit(3).h(0).cx(0, 2)
program = compile(
    circuit,
    coupling_map=((0, 1), (1, 2)),
    routing_strategy="restore_after_each_gate",
)
assert program.n_wires == 3
```

The migration target is the next minor release, proposed as 0.5.0, subject to
release-owner confirmation. The old explicit strategy is not deprecated and
has no planned removal version. Release notes must announce the changed
implicit behavior and the explicit compatibility spelling. The API owner must
confirm whether the release policy instead requires a longer transition before
implementation; this proposal does not override that policy.

## Alternatives and scope

- Keep the default and document `auto`: compatible, but leaves the ordinary
  topology workflow dependent on expert discovery.
- Default directly to SABRE: rejected because a fixed strategy bypasses
  calibration restrictions and can lose on workloads where another plan costs
  less under the existing objective.
- Enable a free final layout: separate deployment and measurement contract
  work; not required for this change.
- Add a second auto selector or a new public options object: unnecessary.
- Add native target admission to root `fq.compile`: a separate accepted-input
  and failure-behavior proposal is required. Root signatures and native target
  rejection are unchanged by this proposal.

Automatic search can cost more compile time. The existing selector plans SABRE
for estimation and routing may plan it again. Before merging, measure that cost
under process limits and investigate implementation reuse if material, without
changing the public selection contract. Runtime or remote execution is not
invoked by compilation.

## Acceptance and approval

The user approved this exact default change on 2026-11-18 in response to the
review question describing the before/after values and preservation of explicit
old-policy calls and layout contracts. Before merge:

1. Prove omitted strategy equals explicit `auto`, and explicit old-policy calls
   retain their behavior.
2. Cover no-topology fast paths, weighted and directed devices, disconnected
   topology failure, symbolic parameters, barriers, and final layout evidence.
3. Preserve deployment replay and conformance tests; do not update snapshots
   simply to accept a test failure.
4. Record compile time and memory against explicit policies on held-out tasks.
   Apply unchanged SDKBench correctness and resource checks. Report losses as
   well as gains; no all-workload improvement is required or assumed.
5. Run focused routing tests, the default smoke/unit tier, and local integration.
   Update documentation, release notes, and the precise approved API contract in
   a separately authorized integration change before implementations land.

The machine-readable authorization is
`contracts/compiler-routing-default-v1-candidate.json`. Existing root export and
historical API manifests do not describe this expert compiler keyword default.
The unapproved routing-plan v2 proposal governs different placement semantics;
amending it would conflate this narrow authorization with an unapproved schema
change. This small record introduces no runtime contract type or public schema.

Approval record: user authorization granted for the exact default change above.
Release version confirmation, CODEOWNER approval, and release gates remain
pending. The requested name `auto` already exists and directly states automatic
strategy selection; no new public name is introduced.
