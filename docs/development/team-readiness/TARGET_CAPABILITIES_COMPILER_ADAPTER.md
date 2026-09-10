# Target Capabilities v1 Compiler Adapter Handoff

Status: minimal internal Compiler adapter; no changes to Stable API, old schemas,
default paths, or failure stages.

## Boundary

The adapter projects private `_compiler.TargetCapabilities` into Core-owned
`RequirementSet` only. Compiler derives requirements from program/target legality;
it does not discover devices, create platform facts, or certify execution evidence.
Production code therefore exposes no Compiler-to-TargetCapabilitySnapshot constructor.

Direct projections cover target class, logical/physical qubit capacity,
measurement results, artifact profiles, shot/operation limits, and maximum
Compiler ancillas. All fallback authorization defaults remain false. The adapter
does not choose targets, execute fallback, or change Runtime behavior.

## Information Loss and Existing Authorities

Fields outside the Core v1 closed vocabulary always produce typed loss records and
remain governed by `compare_target_capabilities()`: topology, control flow,
mid-circuit measurements, reset, timing, pulses, noise, parameter binding, and
calibration hash/validity. Records remain even for default/inactive fields so
migration does not confuse an absent requirement with an absent field.

Parameter-domain coverage in `gates.native` and ranked `ancillas.policy` coverage
are richer than Core v1's generic `covers`. They do not enter an independently
matchable Core RequirementSet; they produce `legacy_comparator_required` records.
`requires_legacy_comparator` explicitly states that the Core subset is not a
complete legality decision. `compare_available()` invokes the unchanged old
comparator. This prevents conservative generic comparisons from rejecting valid
broader gate parameter domains or higher ancilla levels, while retaining requirements.

Adapter results preserve canonical JSON of the old semantic payload and the
original fingerprint. They can restore old objects and verify unchanged
fingerprints. `display_label` remains nonsemantic and is excluded from preserved
semantic payloads.

## Lifecycle

- Owner: Compiler.
- Adapter version: 1.0.
- Supported legacy schema: `target_capabilities_v1`.
- Observability: typed loss count, active loss count, and Core/legacy comparison divergence.
- Exit: all Compiler consumers use approved Core contracts, old importer usage
  reaches zero, and golden schema/fingerprint/compatibility fixtures keep passing.
  Integration must separately approve the removal version.

This change does not alter old TargetCapabilities, semantic_fingerprint,
comparators, defaults, public exports, deployment/Runtime call paths, or historical
contracts.
