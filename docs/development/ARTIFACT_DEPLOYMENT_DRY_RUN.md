# Executable artifact deployment dry run

Updated: 2026-09-10

Status: **implemented as an internal, side-effect-free adapter**

## Decision

Deployment may prepare a verified `ProgramArtifactV2` or `ProgramArtifactV3`
executable for a resolved remote target only after Runtime compatibility preflight
succeeds. Version 3 additionally requires its source artifact and matching
compilation-evidence 3.0 bundle. The dry run
binds the execution request's provider, target and shot count to the exact
artifact and capability snapshot without changing either identity.

This is not a conversion to the legacy `DeploymentPackage`. That package mixes
program text, `CircuitIR`, shots, provider metadata and routing evidence under a
different identity model. Fabricating its missing `CircuitIR` or reusing its
digest would create false evidence.

## Boundary

`prepare_artifact_deployment()`:

- requires the requested provider and target to match the supplied capability
  snapshot exactly;
- delegates artifact, profile, snapshot and shot compatibility to Runtime;
- returns the original executable text and identities without rewriting them;
- exposes v3 logical result width and compilation-local physical result slots
  without treating them as provider qubit identifiers;
- does not read credentials, contact a provider, create a task or claim that a
provider accepted the program;
- is internal and does not change the stable `flagquantum.deployment` exports.

The next provider-specific step must consume this prepared handoff behind the
Remote boundary and produce a separate submission receipt. It must not add job
state, credentials or provider locators to either artifact version.
