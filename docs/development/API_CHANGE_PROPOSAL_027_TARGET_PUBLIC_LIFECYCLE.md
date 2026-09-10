# API Change Proposal 027: Target capability public lifecycle

## Status

**Proposed on 2026-09-10; owner approval required before implementation.**

Approval token:
`approve API_CHANGE_PROPOSAL_027_TARGET_PUBLIC_LIFECYCLE`

## Problem

Artifact compilation, Runtime preflight, evidence verification, and Deployment
dry-run currently require the internal Core `TargetCapabilitySnapshot`. Proposal
026 deliberately did not expose those workflows because users have no reviewed
public way to load, inspect, or match the target facts bound into artifacts.

The Core model already provides immutable snapshots, immutable requirement sets,
closed capability names, evidence levels, validity intervals, canonical identities,
and a pure matcher. Directly exporting its concrete dataclasses would nevertheless
make the entire construction vocabulary and every nested implementation class a
public commitment. Its current `from_json` helpers also use ordinary `json.loads`,
so a public network-facing boundary still needs an input-size limit and duplicate-
key rejection. Finally, matching with an implicit current time is unsuitable for
reproducible public evidence.

## Decision

Introduce a read-only preview at `flagquantum.experimental.targets` with exactly:

```text
TargetSnapshot
RequirementSet
CapabilityMatch
load_target_snapshot
dump_target_snapshot
load_requirement_set
dump_requirement_set
match_capabilities
```

The first three names are frozen branded role views, not aliases or subclasses of
Core dataclasses. Loaders accept bounded JSON text, reject duplicate object keys,
delegate closed-schema and identity validation to Core, and return the matching
view. Dumpers accept only the exact branded view and return the existing Core
canonical JSON without identity changes.

`match_capabilities(requirements, snapshot, *, evaluated_at)` accepts only the two
public views and requires an explicit timezone-aware `datetime`. It delegates to
the pure Core matcher and returns a frozen `CapabilityMatch` view. It performs no
discovery, refresh, fallback selection, compilation, execution, or provider call.

## View surface

`TargetSnapshot` exposes:

```text
version
identity
target_id
target_class
provider
captured_at
valid_until
capability_names
fact(name) -> frozen mapping | null
blockers -> tuple of frozen mappings
to_dict()
to_json()
```

`fact(name)` accepts only a name in the current closed capability vocabulary. An
absent fact returns `None`; unknown names raise `ValueError`. The returned mapping
retains support status, exposure, source, evidence level, value, scope, and blockers
from the identity-validated snapshot. It does not collapse unknown, unmeasured,
unsupported, not-exposed, and absent states into one boolean.

`RequirementSet` exposes only `version`, `identity`, requirement count,
`to_dict()`, and `to_json()`. Construction and mutation remain excluded.

`CapabilityMatch` exposes `executable`, immutable blocker mappings, satisfied and
total preference counts, the requirement-set identity, snapshot identity, and the
explicit evaluation timestamp. The last three values are bound by the preview
adapter so a detached match result cannot be mistaken for another evaluation.

## Time, identity, and scope semantics

The matcher must reject naive datetimes. It normalizes an aware `evaluated_at` to
UTC and records the canonical timestamp in `CapabilityMatch`. Snapshot validity is
evaluated only at that supplied instant.

Target, environment, and scoped device identifiers already validated by the
snapshot remain inspectable. They are not credentials. A physical slot in a
Compiler plan is not inferred to be any snapshot device identifier or provider
qubit. Such binding remains a separately reviewed contract.

The first preview exposes only matching at the snapshot's recorded target identity
and scope. Overrides for expected identity, required scope, minimum claim evidence,
extension handlers, or fallback policy are not public in this version.

## Input and security boundary

- Snapshot and requirement JSON input is limited to 16 MiB before decoding.
- Duplicate keys at any nesting level fail before Core construction.
- Input is text only, never a path, URL, bytes, stream, provider SDK object, or
  callback.
- Canonical Core schema, version, vocabulary, finite-value, timestamp, extension-
  namespace, evidence, and identity checks remain authoritative.
- Load and match do not dereference evidence IDs, access credentials, refresh
  calibration, query devices, or contact a provider.
- Views retain immutable Core values and return detached dictionaries or immutable
  mappings; they do not create a second target-capability authority.

## Lifecycle

1. `experimental_read_only_match`: this proposal.
2. `stable_read_only_candidate`: requires usage, privacy, vocabulary, typing,
   error, extension, and compatibility evidence plus separate approval.
3. `stable_discovery_candidate`: provider discovery, refresh, caching, trust, and
   evidence retrieval require a separate provider-facing proposal.
4. `stable_compilation_candidate`: artifact compilation and verification become
   eligible for proposal only after the read-only target interface is proven.

Experimental rename or removal requires release-note notice and one minor-release
compatibility alias after the preview is published. No lifecycle transition changes
snapshot JSON or identity semantics.

## Compatibility

- Core target-capability version 1.0 bytes, identity, vocabulary, and matcher remain
  unchanged.
- Proposal 026 artifact preview remains unchanged.
- `flagquantum.experimental.__all__` gains only `targets` after approval; the eight
  symbols are not re-exported from the experimental package or stable root.
- Stable package exports, public API snapshot, default compilation, execution,
  Runtime, Remote, and Deployment behavior remain unchanged.

## Excluded scope

This proposal does not authorize snapshot or requirement construction, mutable
facts, capability discovery, calibration refresh, provider SDKs, credentials,
network access, caching, trust stores, signature verification, physical-slot to
provider-qubit binding, artifact compilation, Runtime verification, Deployment,
submission, receipts, jobs, results, QEC/FTOC semantics, or stable promotion.

## Implementation gates

1. **Complete:** record this proposal and exact machine-readable candidate.
2. Receive the exact approval token from the API owner.
3. Add the lazy `experimental.targets` domain with only the eight approved names.
4. Implement frozen role views, bounded duplicate-safe loaders, exact dumpers, and
   detached/immutable inspection results.
5. Implement deterministic explicit-time matching and bind its input identities and
   evaluation timestamp into the match view.
6. Test canonical round trips, identity preservation, all fact states, stale and
   not-yet-valid snapshots, preference counts, malformed/duplicate/unknown/oversized
   input, naive time, lookalikes, laziness, and stable API preservation.
7. Publish preview documentation and keep discovery, workflow, provider, and stable
   promotion behind later proposals.

## Acceptance

- Users can load, inspect, and canonically dump an identity-validated snapshot and
  requirement set without importing internal Core classes.
- Fact inspection preserves the distinction among absence and every support or
  exposure state.
- Matching is deterministic for explicit inputs and time and reports immutable
  blockers and preference counts with bound identities.
- Malformed, duplicate, unknown, oversized, tampered, stale, future, naive-time, and
  lookalike cases fail closed or return the exact Core blocker semantics.
- Import remains lazy; stable exports and default behavior do not change.
- No discovery, physical binding, workflow, provider, or QEC/FTOC capability becomes
  public by implication.

## Approval required

Implementation may begin only after the API owner supplies exactly:

`approve API_CHANGE_PROPOSAL_027_TARGET_PUBLIC_LIFECYCLE`
