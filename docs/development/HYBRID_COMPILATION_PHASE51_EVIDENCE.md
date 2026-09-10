# Hybrid compilation Phase 51 evidence

Date: 2026-09-10

Status: **complete — approved experimental artifact read-only preview**

Authorization:
`approve API_CHANGE_PROPOSAL_026_ARTIFACT_PUBLIC_LIFECYCLE`

## Delivered boundary

The new lazy domain `flagquantum.experimental.artifacts` contains exactly the six
approved symbols. Two frozen branded views provide version-neutral artifact and
evidence roles; four load/dump functions delegate parsing, validation, version
dispatch, identity verification, and canonical encoding to Core.

Program artifact versions 1, 2, and 3 and compilation-evidence versions 1, 2, and
3 round-trip without exposing their concrete classes as the public contract. The
artifact view normalizes the legacy `content_hash` and newer `artifact_identity`
storage names behind one read-only `identity` property.

## Preserved limits

- Input and output are bounded JSON text only.
- Dumpers reject underlying internal objects, mappings, and lookalikes.
- No constructor, binder, Compiler, Runtime, Deployment, provider, or QEC/FTOC
  operation is exposed.
- Stable root and package exports and default paths are unchanged.
- Stable promotion remains separately gated.

## Verification

Tests compile real evidence 1.0/2.0/3.0 and artifacts 2.0/3.0, add a legacy v1
artifact, and verify canonical round trips, identity preservation, frozen views,
lookalike rejection, malformed/duplicate/unsupported/oversized failure, lazy
imports, exact preview exports, and the retained stable public API snapshot.
