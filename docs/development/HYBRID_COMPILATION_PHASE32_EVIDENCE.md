# Phase 32 ProgramArtifact-v2 Core implementation evidence

Date: 2026-09-10

## Authorization

The API owner supplied the exact approval token
`approve API_CHANGE_PROPOSAL_022_PROGRAM_ARTIFACT_V2` after the Phase 31.1
circuit-profile revision. This authorizes the Core portion of Proposal 022.

## Implemented scope

- strict `ProgramArtifactV2` circuit and executable-text profiles;
- exact profile registry for CircuitIR, OpenQASM 2, OpenQASM 3, and QCIS 1;
- canonical payload digest, final-circuit hash, and envelope identity checks;
- explicit v1/v2 dictionary and duplicate-key-safe JSON dispatch;
- circuit construction with parameter-schema derivation;
- executable validation against Core `RequirementSet` and the complete target,
  compilation, binding, and result records;
- 16 MiB payload, 18 MiB envelope, depth, aggregate-entry, and ordinary-string
  limits plus prohibited structured locator/secret fields;
- explicit, fail-closed migration of only unambiguous v1 circuit artifacts.

## Compatibility evidence

The existing `ProgramArtifact` implementation was not edited. The pinned v1
fixture still serializes identically and retains content hash
`0e78aa7d52bce7ff4b1485b11054266bbacc996ea38a4ba1446375d316df517c`.
The v2 circuit constructor reproduces the pinned v2 artifact identity
`b2ec6616b54a0a97241b5b2c7bdf40fd4a698219eca1f5e7d594664322e99292`.

## Verification

The focused Core suite covers v1 compatibility, v2 golden round trips,
symbolic parameters, all identity-tamper classes, exact field shape, duplicate
JSON keys, executable semantics, limits, sensitive structured data, and every
v1 migration blocker. The combined hybrid-compiler, Core-v2, proposal, and
private-contract suite passed **241 tests**. All repository pre-commit checks
also passed before the phase commit.

## Explicit non-scope

No Compiler construction adapter, Runtime consumer, Deployment/provider path,
public export, default-path change, numerical execution, or performance claim
is introduced by Phase 32.
