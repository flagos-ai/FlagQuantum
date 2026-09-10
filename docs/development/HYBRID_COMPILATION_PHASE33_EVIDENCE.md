# Phase 33 Artifact-v2 vertical slice and local execution evidence

Date: 2026-09-10

## Result

The approved ProgramArtifact-v2 contract now has bounded internal adapters for
verified Compiler construction, Runtime target preflight, Deployment dry-run,
and canonical local execution of fully bound circuit artifacts.

`execute_circuit_artifact` accepts only `kind="circuit"` with the exact
`circuit-ir-1.0` profile. It reconstructs the strict Core `CircuitIR`, checks
its content identity, invokes the existing Runtime planner and executor, and
adds artifact identities to the normal `ExecutionResult` provenance and
compatibility records.

## Verified behavior

- a serialized and re-read Bell circuit artifact executes locally;
- the Runtime result contains samples and the complete artifact identity link;
- different shot requests retain the same artifact identity;
- v1 artifacts fail rather than being migrated implicitly;
- symbolic circuit artifacts fail until an explicit binding contract exists;
- OpenQASM and QCIS executable artifacts fail local simulation and remain
  target-bound deployment products;
- no provider call, credentials, public export, or new numerical path is used.

The combined hybrid compiler, Core artifact, Compiler/Runtime/Deployment
adapter, local artifact execution, and private-contract suite passed **257
tests**.

## Claim boundary

This is an internal vertical slice. It does not make ProgramArtifact v2 a
stable public API, execute a target-text artifact locally, submit to a provider,
or claim compilation or execution performance.
