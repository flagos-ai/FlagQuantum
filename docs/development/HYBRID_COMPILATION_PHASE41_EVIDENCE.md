# Phase 41 directional evidence Runtime and Deployment handoff

Date: 2026-09-10

## Result

Proposal 024 is complete. Runtime accepts either compilation-evidence 1.0 or
2.0 and verifies the bundle against the actual fully bound source artifact or
binding result, actual target-capability snapshot, and actual executable
artifact. Deployment can carry the verified version 2.0 bundle beside the
executable artifact in its existing side-effect-free dry-run handoff.

The adapters do not reconstruct Compiler objects, rewrite evidence, repair a
layout, reinterpret physical instructions, select a target, access credentials,
or submit provider work. Core remains authoritative for structural and identity
validation; Runtime binds that evidence to the concrete handoff objects.

## Verified vertical slice

```text
CircuitIR artifact
  -> directed placement/routing
  -> native and direction legalization
  -> physical plan 2.0
  -> executable OpenQASM 3 artifact
  -> compilation-evidence bundle 2.0
  -> canonical JSON round trip
  -> Runtime source/target/output verification
  -> Deployment dry-run carrying the same immutable bundle
```

The test proves that the decoded bundle retains the Compiler plan identity and
that Deployment exposes the same bundle identity. Replacing the actual source
artifact fails Runtime lineage validation.

The combined hybrid-compiler, Core evidence 1.0/2.0, artifact compilation,
Runtime/Deployment handoff, proposal, and private-contract suite passes 304
tests.

## Compatibility and claim boundary

Version 1.0 handoff tests remain unchanged and pass beside version 2.0. Neither
ProgramArtifact version changes. No public root export, default-path change,
provider submission, credential access, job lifecycle state, physical ancilla
allocation, fault-tolerant expansion, or hardware-performance claim is added.
