# Phase 31.1 ProgramArtifact-v2 circuit-profile revision evidence

Date: 2026-09-10

## Result

Proposal 022 now covers both the repository's demonstrated v1 circuit use case
and the Phase 29/30 executable-text path. The added `circuit-ir-1.0` profile
uses a strict Core CircuitIR mapping and the same v2 envelope identity rules.

After approval, new writes are proposed to use v2 exclusively while v1 remains
a read-only compatibility format. Automatic migration is limited to v1 circuit
artifacts with no metadata, flat capability hints, or opaque parent hashes and
with a payload accepted by `CircuitIR.from_dict()`.

## Verification summary

| Gate | Result |
| --- | --- |
| v1 fixture payload passes strict CircuitIR reconstruction | pass |
| Corrected v1 envelope hash is pinned | pass |
| v2 circuit payload hash equals CircuitIR content hash | pass |
| v2 circuit envelope identity independently recomputes | pass |
| Circuit and three executable profiles are closed | pass |
| Circuit target, compilation, requirements, and result fields are null | pass |
| New-write-v2 and v1-read-only policy is machine checked | pass |
| Unsafe or ambiguous v1 migration remains fail closed | pass |
| Phase 31.1 candidate and migration-contract tests | 9 passed |
| Hybrid compiler, proposal, and private-contract focused suite | 225 passed |

## Claim boundary

This is still an unapproved proposal revision. No ProgramArtifact v2 code,
v1 migrator, Core behavior, Compiler adapter, Runtime adapter, Deployment path,
public API, provider call, default-path change, or performance claim is added.
