# Hybrid compilation Phase 48 evidence

Date: 2026-09-10

Status: **complete — approved Deployment v3 dry-run slice**

Authorization:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

## Delivered boundary

Deployment's internal artifact dry run now accepts ProgramArtifact v3 only when
the caller supplies its source artifact and matching compilation-evidence 3.0
bundle. Runtime performs capability, shot, target, lineage, plan, allocation, and
result-projection validation before Deployment creates the immutable handoff.

The handoff retains the exact artifact and evidence objects. Its result view exposes
the logical result width and the compilation-local physical result slots authenticated
by the artifact/evidence pair. Version-2 handoffs preserve their existing optional-
evidence behavior and report no physical result slots.

## Side-effect and identifier boundary

The dry run reads no credentials, performs no network operation, creates no task,
and records no provider response. Physical result slots remain dense indices in the
selected compilation snapshot. They are not provider qubit identifiers and the
Deployment layer does not translate, infer, or enrich them.

## Verification

Both allocated OpenQASM profiles pass the complete Compiler → Core → Runtime →
Deployment dry-run chain. Tests require v3 evidence, preserve object identity,
verify logical width and physical result slots, and confirm that no credential or
task fields appear. The v1/v2 evidence, artifact, Runtime, and Deployment suites
remain unchanged and green.
