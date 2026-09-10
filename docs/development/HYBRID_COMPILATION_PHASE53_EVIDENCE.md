# Hybrid compilation Phase 53 evidence

Date: 2026-09-10

Status: **complete — approved experimental target read-only match preview**

Authorization:
`approve API_CHANGE_PROPOSAL_027_TARGET_PUBLIC_LIFECYCLE`

## Delivered boundary

The lazy `flagquantum.experimental.targets` domain contains exactly the eight
approved symbols. Frozen branded views cover snapshots, requirement sets, and
match results without making Core value classes the public type hierarchy.

Snapshot and requirement loaders accept at most 16 MiB of UTF-8 JSON text, reject
duplicate keys at every nesting level, and delegate canonical identity and
semantic validation to Core. Dumpers preserve exact canonical Core JSON. Fact
inspection returns immutable mappings and does not collapse absent, unknown,
unmeasured, unsupported, or exposure states.

The pure matcher requires a timezone-aware `evaluated_at`, normalizes it to a
canonical UTC timestamp, and binds that timestamp and both input identities into
the result view. It never reads the wall clock through the public call.

## Preserved limits

- No snapshot or requirement construction, discovery, refresh, caching, provider
  SDK, credential, network, or evidence dereference API is exposed.
- No physical-slot/provider-qubit binding or artifact compilation and verification
  workflow is exposed.
- Stable root exports and default execution behavior are unchanged.
- Runtime, Deployment, submission, QEC/FTOC semantics, and stable promotion remain
  separately gated.

## Verification

Tests cover exact exports, lazy import behavior, canonical round trips, identity
preservation, immutable and detached views, fact-state distinction, preference
counts, stale snapshots, explicit timezone normalization, duplicate/non-object/
oversized input, identity tampering, unknown fields, lookalikes, and stable API
preservation.
