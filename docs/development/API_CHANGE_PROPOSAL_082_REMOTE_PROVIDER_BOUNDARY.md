# Proposal 082: Service-owned Remote adapters and neutral-atom provider retirement

## Status and decision scope

**Concrete migration contract awaiting approval.** On 2026-10-22 the repository
owner instructed this boundary correction after reviewing PR #702. That direction
authorizes this proposal and review preparation; the exact removals below are
presented before implementation. Baseline: main
`4972fd0ce9b03a809cca78dc298810bb45a3c5c0`, including Core #700 and Deployment #701.

The affected Remote names are absent from the stable and experimental API
snapshots. Under [PUBLIC_API_POLICY](../reference/PUBLIC_API_POLICY.md) they are
internal APIs, so this proposal does not invent a Stable Core compatibility
promise or require permanent aliases. A separate contract is still appropriate:
the change revises proposal 081's approved Remote-consumer delivery gate and the
checked provider-roster capability claim. No snapshot or maturity record is
regenerated to authorize the change. Document merge alone is not implementation
approval, and implementation approval is not a blanket PR merge grant.

Approve these four decisions together:

1. Remove the generic, non-submitting NeutralAtomProvider and its Remote module
   and exports, without a compatibility shim or replacement provider shell.
2. Keep existing Core geometry and RydbergProgram semantics; do not move vendor
   declaration parsing or CloudBackendProfile construction into Core/Simulation.
3. Correct current provider-roster evidence and examples, explicitly recording
   the missing neutral-atom service integration rather than counting a refusal
   adapter as an implemented cloud provider.
4. Replace proposal 081's generic Remote-coordinate-consumer gate with actual
   service-owned adapter consumption when a concrete integration is delivered.
   Preserve its independent native Simulation, physics, gradient and replay gates.

## Problem and ownership

A physics route is not an external task service. The existing
`remote/qpu/neutral_atom.py` parses a caller-provided dictionary, derives a radius
graph and builds a provider="neutral-atom" profile, but its submit method always
raises and its dry run always reports an analog-schedule blocker. No endpoint,
credentials, device identity, task identifier or result protocol exists behind
that provider name. It should not be the architecture for QuEra, Pasqal, or every
future neutral-atom machine.

Core owns mathematical geometry, atom/site identity, typed physical controls and
source validation. Simulation owns numerical model assembly and evolution.
Deployment owns target-neutral packaging. Remote owns adapters for actual
external services and their protocols. Discovery-only adapters may exist when
there is a concrete external service and an accurately bounded discovery path;
retirement here is about the generic physics-route placeholder, not a rule that
all providers must immediately support every execution mode.

Use existing `remote/qpu/braket.py` for future Braket AHS work when the concrete
contract is approved; do not add a second Braket transport or an empty QuEra
package. A future direct Pasqal integration belongs at its actual Remote service
boundary. No SDK dependency, credential flow, network operation or cloud adapter
is implemented or approved by this migration.

## Exact removal and retained behavior

Delete `flagquantum/remote/qpu/neutral_atom.py`. Its exported names are:

- NEUTRAL_ATOM_PROVIDER, REGISTER_NAMES and REGISTER_UNITS;
- NeutralAtomProvider;
- neutral_atom_sites, interaction_edges and neutral_atom_backend_profile.

Remove NeutralAtomProvider, interaction_edges and neutral_atom_backend_profile
from both `flagquantum.remote.qpu` and `flagquantum.remote` imports and __all__.
Do not leave aliases, a stub module, dynamic fallback, deprecation wrapper or a
new generic register/backend registry. Callers relying on the internal import
must migrate in the same implementation PR; there is no promised old payload
reader for these profiles.

Keep Core's finite_real/cartesian_pair and experimental RydbergProgram unchanged.
The generic provider="neutral-atom" CloudBackendProfile constructor is retired,
not renamed into Core. A distance-radius graph is not an AHS capability schema
or a device calibration. Existing circuit providers, their profiles, package
schemas, submission handles and result protocols stay unchanged. No CircuitIR
or Lindblad plan format changes in this migration.

The repository search found the generic provider's consumers in roster tests,
its offline example, parity evidence and approval records. Move those current
user examples toward the existing native Core construction example below;
retain tests of other providers and their real refusals. Historical proposal
and authorization records remain historical: append a supersession note instead
of silently rewriting what was approved or erasing original evidence.

## Runnable current migration example

For local atom geometry, use the already implemented Core-backed frontend. This
runs on the inspected main without credentials, hardware, or numerical evolution:

```python
from flagquantum.experimental.simulation import RydbergProgram

program = RydbergProgram(
    atoms=(("a", "left"), ("b", "near")),
    sites={"left": (0.0, 0.0), "near": (4.0, 0.0), "far": (8.0, 0.0)},
    length_unit="um", time_unit="us", c6=64.0, loss_rate=0.2,
    minimum_separation=1.0, transport_channels=("aod",),
)
program.move("b", source="near", destination="far",
             start=0.0, duration=1.0, channel="aod")
program.to_ir()
print(program.atom_ids)
print(program.commanded_position("b", 0.5))
print(program.dimensions)
```

Expected:

```text
('a', 'b')
(6.0, 0.0)
(3, 3)
```

This replaces local physical-program construction, not a cloud device discovery
or submission API. There is currently no FlagQuantum replacement submission call
for the generic provider, because it never implemented submission. Concrete
service adapters must publish their own executable workflow and exact refusals.

## Delivery and acceptance

1. Review and explicitly approve the four decisions above. Keep #702 draft while
   deciding the successor boundary; its checks may finish without cancellation.
   If this retirement is approved, supersede #702 instead of merging an unused
   intermediate adapter refactor. Core's shared primitive is already on main.
2. Deliver the retirement as a Remote-owned PR plus shared consumers/evidence.
   Remove the module and both namespace exports. No private generic replacement
   or duplicate geometry/solver layer is allowed.
3. Remove only consumers of the retired generic API. Keep other provider roster,
   declaration errors, package/routing, remote job lifecycle and native-input
   refusal assertions. Add negative import/export tests and runnable Core-native
   construction coverage; a provider-count change alone is insufficient.
4. Update the authoritative cudaq-parity-matrix.toml row, generated parity docs
   and hardware_provider_roster evidence together. Distinguish service adapters,
   generic transport adapters, discovery-only support and actual execution.
   Keep maturity partial and no-live-hardware-evidence claims. Do not relabel the
   remaining seven adapters as seven hardware-tested submitting services.
5. Append the approved 081 gate supersession to its authorization record and
   roadmap. Retain original old-plan/source compatibility captures and all native
   physics, gradient and numerical replay consumers unchanged.
6. Run ownership, architecture, API, parity/maturity and documentation gates,
   supported-Python focused consumers, the required default tier, and latest full
   applicable CI/GPU. Each implementation PR needs its own merge authorization.

A future Braket AHS or Pasqal adapter requires the concrete source-to-provider
mapping, units, supported controls, unsupported-move refusal, authentication,
submit/poll/cancel/results, optional SDK isolation and task evidence. An offline
transport test is not a live QPU execution claim. No such integration, native
numerical execution or complete M3 acceptance is established by this document.
